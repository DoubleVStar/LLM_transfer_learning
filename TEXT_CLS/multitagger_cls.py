#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import io
import json
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ------------------------------
# Data loading helpers (simplified)
# ------------------------------

def read_table(path: str) -> pd.DataFrame:
    """Read CSV/TSV/JSONL (optionally .gz) with robust encoding fallback.
    Supports: .csv, .csv.gz, .tsv, .tsv.gz, .jsonl, .jsonl.gz, .ndjson
    """
    pl = path.lower()
    encodings_try = [None, "utf-8", "utf-8-sig", "cp1252", "latin-1"]

    def _read_csv(p, **kw):
        last = None
        for enc in encodings_try:
            try:
                return pd.read_csv(p, encoding=enc, **kw)
            except Exception as e:
                last = e
        raise last

    if pl.endswith((".csv", ".csv.gz")):
        return _read_csv(path)
    if pl.endswith((".tsv", ".tsv.gz")):
        return _read_csv(path, sep="\t")
    if pl.endswith((".jsonl", ".jsonl.gz", ".ndjson")):
        return pd.read_json(path, lines=True)
    ext = os.path.splitext(path)[1].lower()
    raise ValueError(f"Unsupported file extension: {ext}. Use .csv / .tsv / .jsonl (+ optional .gz)")


def guess_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Heuristically guess (text_col, label_col). Raises if not found."""
    text_cands = ["text", "TEXT", "abstract", "ABSTRACT", "title", "TITLE", "body", "Body"]
    label_cands = ["label", "LABEL", "study_type", "category", "class", "target", "y"]
    text_col = next((c for c in text_cands if c in df.columns), None)
    label_col = next((c for c in label_cands if c in df.columns), None)
    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        text_col = obj_cols[0] if obj_cols else None
    if text_col is None or label_col is None:
        raise ValueError(
            f"Unable to guess columns. Available columns: {list(df.columns)}. "
            "Please pass --text_col and --label_col explicitly."
        )
    return text_col, label_col

# ------------------------------
# Metrics
# ------------------------------

def build_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        # Robust to both tuple (preds, labels) and EvalPrediction objects
        try:
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        except AttributeError:
            logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        y_true = labels
        y_pred = preds
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }
    return compute_metrics

# ------------------------------
# Dataset
# ------------------------------

class CLSDataset(Dataset):
    """Simple classification dataset for Hugging Face Trainer."""
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str, label2id: Dict[str, int], max_length: int = 256):
        
        assert text_col in df.columns and label_col in df.columns, f"Missing columns: {text_col}/{label_col}"
        self.texts = df[text_col].astype(str).tolist()
        labels_clean = df[label_col].map(clean_label_value)
        self.labels = [label2id[str(lbl)] for lbl in labels_clean.tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True, padding=False, max_length=self.max_length)
        enc["labels"] = int(self.labels[idx])
        return enc
    



# ------------------------------
# Main
# ------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Classification fine-tuning from a trained NER checkpoint (simplified)")
    ap.add_argument("-n", "--ner_checkpoint", required=True, type=str, help="Path to NER checkpoint directory")
    ap.add_argument("-tr", "--train_file", type=str, default='/home/kevin/MultiTagger-v2/DATA/TEXT_CLS/data_split/train.csv', help="Training file (.csv/.tsv/.jsonl[.gz])")
    ap.add_argument("-va", "--val_file", type=str, default='/home/kevin/MultiTagger-v2/DATA/TEXT_CLS/data_split/dev.csv', help="Validation file (.csv/.tsv/.jsonl[.gz]); if omitted, 10% of train is used")
    ap.add_argument("-te", "--test_file", type=str, default='/home/kevin/MultiTagger-v2/DATA/TEXT_CLS/data_split/test.csv', help="Optional test file (.csv/.tsv/.jsonl[.gz])")
    ap.add_argument("-x", "--text_col", type=str, default='TEXT', help="Text column name (auto-guess if omitted)")
    ap.add_argument("-y", "--label_col", type=str, default='LABEL', help="Label column name (auto-guess if omitted)")
    ap.add_argument("-o", "--output_dir", required=True, type=str, help="Where to save model & metrics")
    # Minimal training knobs
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--drop_unknown", action="store_true", help="Drop rows in dev/test whose labels are unseen in train")


    return ap.parse_args()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ------------ Label cleaning & validation helpers ------------
def clean_label_value(x: object) -> str:
    """Make a label comparable: strip spaces and surrounding quotes."""
    s = str(x).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s

def clean_label_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].map(clean_label_value)
    return df

def validate_or_filter_labels(df_split: pd.DataFrame, label_col: str, label2id: Dict[str, int],
                              split_name: str, drop_unknown: bool):
    """Ensure dev/test labels ∈ train set; optionally drop unknowns."""
    vals = set(df_split[label_col].astype(str).map(clean_label_value))
    unknown = sorted([v for v in vals if v not in label2id])
    if unknown:
        if drop_unknown:
            before = len(df_split)
            mask = df_split[label_col].map(clean_label_value).isin(label2id.keys())
            dropped = df_split.loc[~mask]
            df_split = df_split.loc[mask].reset_index(drop=True)
            after = len(df_split)
            print(f"[WARN] {split_name}: dropped {before - after} rows with unknown labels "
                  f"(examples: {unknown[:10]}).")
        else:
            raise ValueError(
                f"{split_name} contains labels unseen in train: {unknown[:10]}. "
                f"Use --drop_unknown or fix your label column."
            )
    return df_split

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    # 1) Load data
    df_tr = read_table(args.train_file)
    if args.val_file:
        df_va = read_table(args.val_file)
    else:
        # Simple split 90/10 when no dev file supplied
        from sklearn.model_selection import train_test_split
        df_tr, df_va = train_test_split(df_tr, test_size=0.10, random_state=args.seed, shuffle=True)
    df_te = read_table(args.test_file) if args.test_file else None

    # 2) Guess columns if not provided
    text_col, label_col = (args.text_col, args.label_col)
    if text_col is None or label_col is None:
        tc, lc = guess_columns(df_tr)
        text_col = text_col or tc
        label_col = label_col or lc
        print(f"[INFO] Auto-guessed columns -> text_col='{text_col}', label_col='{label_col}'")

    # Clean labels after deciding the label column
    df_tr = clean_label_column(df_tr, label_col)
    df_va = clean_label_column(df_va, label_col)
    if df_te is not None:
        df_te = clean_label_column(df_te, label_col)


    # 3) Build label mapping from train set

    labels_sorted = sorted(pd.unique(df_tr[label_col].astype(str)))
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    id2label = {i: l for l, i in label2id.items()}


    with io.open(os.path.join(args.output_dir, "labels_cls.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2, ensure_ascii=False)

    # 4) Tokenizer & model from NER checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.ner_checkpoint, use_fast=True)
    config = AutoConfig.from_pretrained(
        args.ner_checkpoint,
        num_labels=len(label2id), id2label=id2label, label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.ner_checkpoint, config=config, ignore_mismatched_sizes=True
    )

    # 5) Datasets
    df_va = validate_or_filter_labels(df_va, label_col, label2id, "dev",  drop_unknown=args.drop_unknown)
    if df_te is not None:
        df_te = validate_or_filter_labels(df_te, label_col, label2id, "test", drop_unknown=args.drop_unknown)

    train_ds = CLSDataset(df_tr, tokenizer, text_col, label_col, label2id, max_length=args.max_length)
    val_ds = CLSDataset(df_va, tokenizer, text_col, label_col, label2id, max_length=args.max_length)
    test_ds = CLSDataset(df_te, tokenizer, text_col, label_col, label2id, max_length=args.max_length) if df_te is not None else None

    # 6) TrainingArguments (keep minimal & version compatible)
    common = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )
    eval_strategy_value = "epoch"
    try:
        training_args = TrainingArguments(eval_strategy=eval_strategy_value, **common)   # transformers >= 4.48+
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy=eval_strategy_value, **common)  # older versions

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(id2label),
    )

    # 8) Train & save
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with io.open(os.path.join(args.output_dir, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)

    # 9) Eval (val + optional test)
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    with io.open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    val_preds = trainer.predict(val_ds)
    y_true = val_preds.label_ids
    y_pred = np.argmax(val_preds.predictions, axis=-1)
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))], zero_division=0)
    with io.open(os.path.join(args.output_dir, "val_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(args.output_dir, "val_confusion_matrix.csv"), cm, delimiter=",", fmt="%d")

    if test_ds is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_ds)
        with io.open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        test_preds = trainer.predict(test_ds)
        y_true_t = test_preds.label_ids
        y_pred_t = np.argmax(test_preds.predictions, axis=-1)
        report_t = classification_report(y_true_t, y_pred_t, target_names=[id2label[i] for i in range(len(id2label))], zero_division=0, digits=4)
        with io.open(os.path.join(args.output_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_t)
        cm_t = confusion_matrix(y_true_t, y_pred_t)
        np.savetxt(os.path.join(args.output_dir, "test_confusion_matrix.csv"), cm_t, delimiter=",", fmt="%d")

    # 10) Save run config (minimal)
    run_cfg = {
        "ner_checkpoint": args.ner_checkpoint,
        "train_file": args.train_file,
        "val_file": args.val_file or "(10% split from train)",
        "test_file": args.test_file,
        "text_col": text_col,
        "label_col": label_col,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    with io.open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2, ensure_ascii=False)

    print("\n[Done] Classification fine-tuning complete.")
    print(f"Best model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

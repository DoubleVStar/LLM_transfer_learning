#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultitaggerV2 — NER training & evaluation script for BIO2 .txt corpora in a folder.

Features
--------
1) Reads BIO2/IOB2 formatted .txt files (token + label per line, blank line between sentences)
2) Trains a token classification (NER) model with Hugging Face Transformers
3) Computes seqeval metrics: precision/recall/F1/accuracy (entity-level)
4) Auto-detects train/dev/test files in the folder; or performs a split if dev is missing
5) Handles subword alignment (labels on first sub-token, rest ignored with -100)
6) Saves label maps and metrics; can resume from checkpoints

Expected folder structure
-------------------------
<data_dir>/
  ├─ train.txt            # required
  ├─ dev.txt | val.txt  # optional (if missing, will split from train)
  └─ test.txt             # optional (if missing, evaluation on dev only)

Each .txt file uses UTF-8 encoding and lines like:
    Aspirin B-Drug
    reduces O
    risk O

    in O
    randomized B-StudyType
    trials I-StudyType

Usage
-----
python multitaggerV2.py \
  --data_dir /home/kevin/MultiTagger-v2/DATA/NER/data_split \
  --output_dir ./outputs/multitagger_v2 \
  --epochs 30 --batch_size 16 --lr 3e-5 --max_length 512

(Optional) Enable gradient accumulation for small GPUs:
  --gradient_accumulation_steps 2 --batch_size 8


Notes
-----
- This script focuses on the NER head of MultitaggerV2. The classification head is
  intentionally disabled (β=0) since input BIO2 files contain token-level labels only.
- All comments are in English as requested previously.
"""

import os
import io
import sys
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

import ner_eval
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

import sys, inspect
import transformers
#print("[DEBUG] python:", sys.executable)
#print("[DEBUG] transformers.version:", transformers.__version__)
#print("[DEBUG] transformers.path:", getattr(transformers, "__file__", transformers.__path__))
#from transformers import TrainingArguments
#print("[DEBUG] TrainingArguments signature:", inspect.signature(TrainingArguments.__init__))
# ------------------------------
# Utilities
# ------------------------------

def set_global_seed(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_bio2_file(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Read a BIO2/IOB2 file.

    Each non-empty line contains: "token<space/tab>label".
    Sentences are separated by blank lines.
    Returns lists of tokens and labels per sentence.
    """
    sentences_tokens: List[List[str]] = []
    sentences_labels: List[List[str]] = []
    tokens: List[str] = []
    labels: List[str] = []

    with io.open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if tokens:
                    sentences_tokens.append(tokens)
                    sentences_labels.append(labels)
                    tokens, labels = [], []
                continue
            # Split by whitespace; last field is label, the rest compose the token if any
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed line in {path}: '{line}' (expect 'TOKEN LABEL')")
            label = parts[-1]
            token = " ".join(parts[:-1])
            tokens.append(token)
            labels.append(label)

    # Flush last sentence if file does not end with blank line
    if tokens:
        sentences_tokens.append(tokens)
        sentences_labels.append(labels)

    # Basic validation
    assert len(sentences_tokens) == len(sentences_labels), "Token/label sentence count mismatch"
    return sentences_tokens, sentences_labels


def load_bio2_from_dir(data_dir: str, dev_size: float = 0.1, seed: int = 42):
    """Load train/dev/test from a directory of BIO2 .txt files.

    Prefers train.txt, (dev.txt|val.txt), test.txt.
    If dev/val is missing: split from train using dev_size.
    If test is missing: evaluation will run on dev only.
    """
    def p(name):
        return os.path.join(data_dir, name)

    train_path = p("train.txt")
    dev_path = p("dev.txt") if os.path.exists(p("dev.txt")) else p("val.txt")
    test_path = p("test.txt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.txt not found in {data_dir}")

    X_train, y_train = read_bio2_file(train_path)

    if os.path.exists(dev_path):
        X_dev, y_dev = read_bio2_file(dev_path)
    else:
        # Create a stratified-ish split by sentence length buckets to keep label distribution roughly similar
        lengths = [len(x) for x in X_train]
        # Bucket by length (0-10, 11-20, ...)
        bins = [min(l // 10, 10) for l in lengths]
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train, y_train, test_size=dev_size, random_state=seed, stratify=bins
        )

    if os.path.exists(test_path):
        X_test, y_test = read_bio2_file(test_path)
    else:
        X_test, y_test = [], []

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)


def build_label_list(train_labels: List[List[str]], dev_labels: List[List[str]], test_labels: List[List[str]]):
    """Build sorted label list with 'O' first, then B-*/I-* in alpha order."""
    uniq = {lab for sent in (train_labels + dev_labels + test_labels) for lab in sent}
    if "O" not in uniq:
        uniq.add("O")
    # Keep 'O' first; others sorted
    others = sorted([l for l in uniq if l != "O"])
    label_list = ["O"] + others
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    return label_list, label2id, id2label


@dataclass
class NERSample:
    tokens: List[str]
    labels: List[str]


class NERDataset(Dataset):
    """A simple PyTorch dataset for word-level NER with subword alignment."""

    def __init__(
        self,
        samples: List[NERSample],
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
        label_all_tokens: bool = False,
    ):
        self.samples = samples
        self.tok = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample.tokens
        labels = sample.labels

        # Tokenize with is_split_into_words=True so tokenizer does not re-split on spaces
        enc = self.tok(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=False,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

        word_ids = enc.word_ids()
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like [CLS], [SEP]
                label_ids.append(-100)
            else:
                label_str = labels[word_idx]
                if word_idx != previous_word_idx:
                    # First sub-token of the word gets the original label
                    label_ids.append(self.label2id[label_str])
                else:
                    # Subsequent sub-tokens: either repeat I- tag or ignore
                    if self.label_all_tokens and label_str.startswith("B-"):
                        label_ids.append(self.label2id["I-" + label_str[2:]])
                    elif self.label_all_tokens and label_str.startswith("I-"):
                        label_ids.append(self.label2id[label_str])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

        enc["labels"] = label_ids
        return enc


# ------------------------------
# Metrics
# ------------------------------

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, id2label: Dict[int, str]):
    """Convert model outputs and true ids to lists of label strings, ignoring -100 positions."""
    preds = np.argmax(predictions, axis=2)
    true_labels: List[List[str]] = []
    true_preds: List[List[str]] = []

    for pred_row, label_row in zip(preds, label_ids):
        cur_pred = []
        cur_true = []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            cur_pred.append(id2label[int(p)])
            cur_true.append(id2label[int(l)])
        true_preds.append(cur_pred)
        true_labels.append(cur_true)
    return true_preds, true_labels


def make_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics_fn(eval_pred):
        predictions, labels = eval_pred
        preds_list, labels_list = align_predictions(predictions, labels, id2label)
        return {
            "precision": precision_score(labels_list, preds_list),
            "recall": recall_score(labels_list, preds_list),
            "f1": f1_score(labels_list, preds_list),
            "accuracy": accuracy_score(labels_list, preds_list),
        }

    return compute_metrics_fn


# ------------------------------
# Main
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MultitaggerV2 NER training/eval for BIO2 data")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing BIO2 train/dev/test .txt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save model & metrics")
    parser.add_argument("--model_name_or_path", type=str, default="allenai/specter2_base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_size", type=float, default=0.1, help="If no dev file, split train by this proportion")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision (Ampere+ GPUs)")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision (A100+/TPU)")
    parser.add_argument("--label_all_tokens", action="store_true", help="Label all sub-tokens (I-*) instead of ignoring")
    parser.add_argument("--eval_on_test", action="store_true", help="Run final evaluation on test.txt if present")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (eval steps)")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_global_seed(args.seed)

    # 1) Load data
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = load_bio2_from_dir(
        args.data_dir, dev_size=args.dev_size, seed=args.seed
    )

    # 2) Build labels
    label_list, label2id, id2label = build_label_list(y_train, y_dev, y_test)

    with io.open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"label_list": label_list, "label2id": label2id}, f, ensure_ascii=False, indent=2)

    # 3) Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)},
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    # 4) Build datasets
    train_samples = [NERSample(t, l) for t, l in zip(X_train, y_train)]
    dev_samples = [NERSample(t, l) for t, l in zip(X_dev, y_dev)]
    test_samples = [NERSample(t, l) for t, l in zip(X_test, y_test)] if X_test else []

    train_ds = NERDataset(train_samples, tokenizer, label2id, max_length=args.max_length, label_all_tokens=args.label_all_tokens)
    dev_ds = NERDataset(dev_samples, tokenizer, label2id, max_length=args.max_length, label_all_tokens=args.label_all_tokens)
    test_ds = NERDataset(test_samples, tokenizer, label2id, max_length=args.max_length, label_all_tokens=args.label_all_tokens) if test_samples else None

    # 5) Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 6) Training args
    logging_steps = args.logging_steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",

        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",  # 或 None
    )
    # 7) Trainer with metrics
    compute_metrics = make_compute_metrics({i: l for i, l in enumerate(label_list)})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Add simple early stopping via callback if patience > 0
    from transformers import EarlyStoppingCallback
    callbacks = []
    if args.patience and args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
        trainer.add_callback(callbacks[-1])

    # 8) Train
    train_result = trainer.train()
    trainer.save_model(args.output_dir)

    # Save train metrics
    with io.open(os.path.join(args.output_dir, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)

    # 9) Evaluate on dev
    dev_metrics = trainer.evaluate(eval_dataset=dev_ds)
    with io.open(os.path.join(args.output_dir, "dev_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=2)

    # Detailed classification report (per-entity)
    preds_dev = trainer.predict(dev_ds)
    preds_list, labels_list = align_predictions(preds_dev.predictions, preds_dev.label_ids, {i: l for i, l in enumerate(label_list)})
    report_dev = classification_report(labels_list, preds_list)
    with io.open(os.path.join(args.output_dir, "dev_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_dev)

    # 10) (Optional) Evaluate on test
    if args.eval_on_test and test_ds is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_ds)
        with io.open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        preds_test = trainer.predict(test_ds)
        preds_list_t, labels_list_t = align_predictions(preds_test.predictions, preds_test.label_ids, {i: l for i, l in enumerate(label_list)})
        report_test = classification_report(labels_list_t, preds_list_t)
        with io.open(os.path.join(args.output_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_test)
        try:
            res = ner_eval.compute_all(labels_list_t, preds_list_t)
            print('ner_eval result: \n', res["strict"]["micro"])  # {'precision': ..., 'recall': ..., 'f1': ...}
        except Exception as e:
            print('ner_eval failed.')

    # 11) Save a small README with run info
    readme = {
        "data_dir": args.data_dir,
        "model_name_or_path": args.model_name_or_path,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "label_all_tokens": args.label_all_tokens,
        "dev_size": args.dev_size,
        "seed": args.seed,
        "notes": "MultitaggerV2 NER-only training over BIO2 data"
    }
    with io.open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(readme, f, indent=2, ensure_ascii=False)

    print("\nTraining & evaluation complete.")
    print(f"Best model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

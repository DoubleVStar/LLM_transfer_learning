#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import inspect
import glob
import os
import random
import json
from typing import List, Dict, Any, Tuple
import ner_eval

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


# import sys, inspect
# import transformers
# print("[DEBUG] python:", sys.executable)
# print("[DEBUG] transformers.version:", transformers.__version__)
# print("[DEBUG] transformers.path:", getattr(transformers, "__file__", transformers.__path__))
# from transformers import TrainingArguments
# print("[DEBUG] TrainingArguments signature:", inspect.signature(TrainingArguments.__init__))


# =========================
# Read BIO2 (IOB2) data
# =========================

def read_bio2_file(path: str, drop_docstart: bool = True) -> List[Dict[str, List[List[str]]]]:
    """
    Read a txt file and return a list of sentences as a "document":
    doc = {"id": <filename>, "tokens": [ [...], ... ], "tags": [ [...], ... ]}

    - Each line: token ... tag (blank line separates sentences)
    - If multiple columns: take the first column as token, the last column as tag
    - Skip lines starting with -DOCSTART-
    """
    tokens, tags = [], []
    all_tokens, all_tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens:
                    all_tokens.append(tokens);
                    all_tags.append(tags)
                    tokens, tags = [], []
                continue
            if drop_docstart and line.startswith("-DOCSTART-"):
                continue
            cols = line.split()
            if len(cols) < 2:
                raise ValueError(f"{path}: invalid line (must have at least 2 columns): {line}")
            token = cols[0]
            tag = cols[-1]
            tokens.append(token)
            tags.append(tag)
    if tokens:
        all_tokens.append(tokens);
        all_tags.append(tags)
    return [{"id": os.path.basename(path), "tokens": all_tokens, "tags": all_tags}]


def read_bio2_glob(pattern: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files found: {pattern}")
    docs = []
    for p in paths:
        docs.extend(read_bio2_file(p))
    return docs


# =========================
# Annotation validation and (optional) fixing
# =========================

def validate_and_optionally_fix_bio2(tokens: List[str], tags: List[str], autofix: bool) -> Tuple[
    List[str], List[str], int]:
    """
    Basic BIO2 validation:
    - Allow O, B-X, I-X
    - If I-X follows O or I/Y (different type), when autofix=True automatically convert to B-X
    Return: tokens, tags_fixed, number of fixes
    """
    fixed = 0
    prev_type = "O"
    prev_in_entity = False
    new_tags = []
    for i, tg in enumerate(tags):
        if tg == "O":
            new_tags.append("O")
            prev_in_entity = False
            prev_type = "O"
            continue
        if "-" not in tg or tg.split("-", 1)[0] not in ("B", "I"):
            raise ValueError(f"Invalid tag (not BIO2): {tg}")
        prefix, etype = tg.split("-", 1)
        if prefix == "B":
            new_tags.append(tg)
            prev_in_entity = True
            prev_type = etype
        else:  # I
            if not prev_in_entity or etype != prev_type:
                # Needs to be converted to B-etype
                if autofix:
                    new_tags.append(f"B-{etype}")
                    fixed += 1
                    prev_in_entity = True
                    prev_type = etype
                else:
                    raise ValueError(
                        f"Invalid I- start (consider --autofix-iob): pos={i}, tag={tg}, prev={new_tags[-1] if new_tags else 'None'}")
            else:
                new_tags.append(tg)
                prev_in_entity = True
                prev_type = etype
    return tokens, new_tags, fixed


def scan_and_fix_corpus(docs: List[Dict[str, Any]], autofix: bool) -> int:
    total_fix = 0
    for d in docs:
        new_tokens, new_tags = [], []
        for toks, tgs in zip(d["tokens"], d["tags"]):
            toks2, tgs2, f = validate_and_optionally_fix_bio2(toks, tgs, autofix)
            new_tokens.append(toks2)
            new_tags.append(tgs2)
            total_fix += f
        d["tokens"], d["tags"] = new_tokens, new_tags
    return total_fix


# =========================
# Dataset splitting
# =========================

def parse_split(split_str: str) -> Tuple[int, int, int]:
    # Support "8,1,1" or "0.8,0.1,0.1"
    parts = [p.strip() for p in split_str.split(",")]
    if len(parts) != 3:
        raise ValueError("--split requires three numbers, e.g. 8,1,1 or 0.8,0.1,0.1")
    vals = [float(p) for p in parts]
    if sum(vals) <= 0:
        raise ValueError("sum of split ratios must be > 0")
    return vals[0], vals[1], vals[2]


def split_docs(docs: List[Dict[str, Any]], split_tuple, seed: int):
    a, b, c = split_tuple
    total = len(docs)
    order = list(range(total))
    random.Random(seed).shuffle(order)
    k = a + b + c
    n_train = int((a / k) * total)
    n_dev = int((b / k) * total)
    n_test = total - n_train - n_dev
    idx_train = order[:n_train]
    idx_dev = order[n_train:n_train + n_dev]
    idx_test = order[n_train + n_dev:]

    def gather(idxs):
        toks, tgs = [], []
        for i in idxs:
            toks.extend(docs[i]["tokens"])
            tgs.extend(docs[i]["tags"])
        return {"tokens": toks, "tags": tgs}

    return gather(idx_train), gather(idx_dev), gather(idx_test)


def split_sentences(docs: List[Dict[str, Any]], split_tuple, seed: int):
    all_sents = []
    for d in docs:
        for toks, tgs in zip(d["tokens"], d["tags"]):
            all_sents.append((toks, tgs))
    total = len(all_sents)
    order = list(range(total))
    random.Random(seed).shuffle(order)

    a, b, c = split_tuple
    k = a + b + c
    n_train = int((a / k) * total)
    n_dev = int((b / k) * total)
    n_test = total - n_train - n_dev

    def gather(idxs):
        toks, tgs = [], []
        for i in idxs:
            s = all_sents[i]
            toks.append(s[0]);
            tgs.append(s[1])
        return {"tokens": toks, "tags": tgs}

    return gather(order[:n_train]), gather(order[n_train:n_train + n_dev]), gather(order[n_train + n_dev:])


def write_conll(split: Dict[str, List[List[str]]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for toks, tgs in zip(split["tokens"], split["tags"]):
            for t, g in zip(toks, tgs):
                f.write(f"{t} {g}\n")
            f.write("\n")


# =========================
# Training (with subword alignment)
# =========================

def collect_labels(*splits) -> List[str]:
    labs = set()
    for sp in splits:
        if sp is None: continue
        for tg_seq in sp["tags"]:
            labs.update(tg_seq)
    labels = sorted(labs - {"O"})
    return ["O"] + labels


def encode_splits(
        data: Dict[str, List[List[str]]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
        label_all_subtokens: bool = False,
) -> Dataset:
    enc = tokenizer(
        data["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    all_labels = []
    for i in range(len(data["tokens"])):
        word_ids = enc.word_ids(batch_index=i)
        seq_tags = data["tags"][i]
        labels = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            else:
                if wid != prev_wid:
                    labels.append(label2id[seq_tags[wid]])
                else:
                    labels.append(label2id[seq_tags[wid]] if label_all_subtokens else -100)
            prev_wid = wid
        all_labels.append(labels)
    return Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": all_labels,
    })


def make_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics(p):
        preds, labels = p
        preds = preds.argmax(-1)
        TL, TP = [], []
        for pr, lb in zip(preds, labels):
            t_seq, p_seq = [], []
            for p_i, l_i in zip(pr, lb):
                if l_i == -100: continue
                t_seq.append(id2label[int(l_i)])
                p_seq.append(id2label[int(p_i)])
            TL.append(t_seq);
            TP.append(p_seq)
        return {
            "precision": precision_score(TL, TP),
            "recall": recall_score(TL, TP),
            "f1": f1_score(TL, TP),
        }

    return compute_metrics


def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # Ampere and above
    except Exception:
        return False


# =========================
# Main workflow
# =========================

def main():
    ap = argparse.ArgumentParser(description="Automatically split multiple BIO2 txt files and train with BioBERT")

    ap.add_argument("--input-glob", required=True,
                    default='/home/kevin/MultiTagger-v2/DATA/cased_bio2_methodology_512/*.txt',
                    help='Input file pattern, e.g. "data/*.txt"')

    ap.add_argument("--out-dir", required=True, help="Output directory (will contain splits and model)")
    ap.add_argument("--split", default="7,1,2", help="train,dev,test ratios (e.g. 8,1,1 or 0.8,0.1,0.1)")
    ap.add_argument("--split-by", choices=["document", "sentence"], default="document",
                    help="Random split by file (document) or by sentence (sentence)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", default="dmis-lab/biobert-base-cased-v1.1")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num-epochs", type=int, default=5)
    ap.add_argument("--label-all-subtokens", action="store_true",
                    help="Propagate word label to all subtokens; default only labels the first subtoken")
    ap.add_argument("--autofix-iob", action="store_true",
                    help="Automatically convert invalid I- starts to B- (otherwise raise error)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Read data
    docs = read_bio2_glob(args.input_glob)
    print(f"[INFO] Number of documents read: {len(docs)}")

    # 2) Validate / fix
    fixed = scan_and_fix_corpus(docs, autofix=args.autofix_iob)
    if fixed > 0:
        print(f"[INFO] Number of automatic IOB2 fixes: {fixed}")

    # 3) Split
    split_tuple = parse_split(args.split)
    if args.split_by == "document":
        train_split, dev_split, test_split = split_docs(docs, split_tuple, args.seed)
    else:
        train_split, dev_split, test_split = split_sentences(docs, split_tuple, args.seed)

    # 4) Export CoNLL (reusable)
    spdir = os.path.join(args.out_dir, "splits")
    os.makedirs(spdir, exist_ok=True)
    train_path = os.path.join(spdir, "train.conll")
    dev_path = os.path.join(spdir, "dev.conll")
    test_path = os.path.join(spdir, "test.conll")
    write_conll(train_split, train_path)
    write_conll(dev_split, dev_path)
    write_conll(test_split, test_path)
    print(f"[INFO] Written: {train_path}, {dev_path}, {test_path}")
    print(
        f"[INFO] Sentence counts | train={len(train_split['tokens'])}, dev={len(dev_split['tokens'])}, test={len(test_split['tokens'])}")

    # 5) Prepare training
    label_list = collect_labels(train_split, dev_split, test_split)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    with open(os.path.join(args.out_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)
    train_ds = encode_splits(train_split, tokenizer, label2id,
                             max_length=args.max_length,
                             label_all_subtokens=args.label_all_subtokens)
    dev_ds = encode_splits(dev_split, tokenizer, label2id,
                           max_length=args.max_length,
                           label_all_subtokens=args.label_all_subtokens)
    test_ds = encode_splits(test_split, tokenizer, label2id,
                            max_length=args.max_length,
                            label_all_subtokens=args.label_all_subtokens)

    # 6) Model
    model = AutoModelForTokenClassification.from_pretrained(
        args.pretrained,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Precision strategy
    use_bf16 = bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # 7) Training
    # training_args = TrainingArguments(
    #     output_dir=os.path.join(args.out_dir, "model"),
    #     learning_rate=args.lr,
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.batch_size,
    #     num_train_epochs=args.num_epochs,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     metric_for_best_model="f1",
    #     greater_is_better=True,
    #     logging_steps=50,
    #     seed=args.seed,
    #     bf16=use_bf16,
    #     fp16=use_fp16,
    #     report_to="none",
    #     save_total_limit=2,
    # )

    ta_kwargs = dict(
        output_dir=os.path.join(args.out_dir, "model"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        logging_steps=50,
        seed=args.seed,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,
        save_total_limit=2,
    )

    # Precision setting (according to variables in your script)
    try:
        ta_kwargs.update(dict(bf16=use_bf16, fp16=use_fp16))
    except Exception:
        ta_kwargs.update(dict(fp16=use_fp16))

    # Compatibility for old/new parameter name: eval_strategy vs evaluation_strategy
    sig = str(inspect.signature(TrainingArguments.__init__))
    if "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**ta_kwargs)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
    )

    trainer.train()
    print("[DEV]", trainer.evaluate())

    # 8) Detailed test report
    preds = trainer.predict(test_ds).predictions.argmax(-1)
    y_true, y_pred = [], []
    for p_seq, l_seq in zip(preds, test_ds["labels"]):
        t_seq, pr_seq = [], []
        for p, l in zip(p_seq, l_seq):
            if l == -100: continue
            t_seq.append(id2label[int(l)])
            pr_seq.append(id2label[int(p)])
        y_true.append(t_seq);
        y_pred.append(pr_seq)

    print("\n=== seqeval classification report (test) ===")
    print(classification_report(y_true, y_pred, digits=4))

    res = ner_eval.compute_all(y_true, y_pred)
    print(res["strict"]["micro"])  # {'precision': ..., 'recall': ..., 'f1': ...}

    # 9) Save final model and tokenizer
    save_dir = os.path.join(args.out_dir, "model_best")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n[OK] Done! Model and label mapping saved to: {save_dir}")
    print(f"[OK] Dataset split files saved in: {spdir}")


if __name__ == "__main__":
    main()

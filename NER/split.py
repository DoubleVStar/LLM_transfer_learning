#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split multiple BIO2/IOB2 .txt documents into train/val/test files by *documents*.

- Input: a folder containing many BIO2 .txt files (each file = one document consisting of sentences).
- Output: three concatenated files in output_dir: train.txt, val.txt, test.txt
  (and a compatibility copy dev.txt == val.txt, since some trainers expect dev.txt).
- Ratio defaults to 0.7 / 0.1 / 0.2 for train / val / test.
- Shuffles documents with a fixed seed for reproducibility.
- Ensures UTF-8 and a blank line between documents in the merged output.
- Writes a manifest JSON listing which source files went to each split.

Usage
-----
python split.py \
  --input_dir /home/kevin/MultiTagger-v2/DATA/cased_bio2_methodology_512 \
  --output_dir /home/kevin/MultiTagger-v2/DATA/NER/data_split \
  --train 0.7 --val 0.1 --test 0.2 --seed 42

Notes
-----
- This script assumes each input .txt is a single document with standard BIO2 format:
  "TOKEN LABEL" per line; empty line separates sentences. We do not parse or validate labels.
- We exclude any existing aggregate files named: train.txt, dev.txt, val.txt, valid.txt, test.txt.
- All comments are in English as requested.
"""

import os
import io
import json
import glob
import math
import random
import argparse
from typing import List, Dict, Tuple

AGGREGATE_BASENAMES = {"train.txt", "dev.txt", "val.txt", "valid.txt", "test.txt"}


def list_documents(input_dir: str, pattern: str = "*.txt") -> List[str]:
    """List candidate BIO2 documents (files) under input_dir matching pattern,
    excluding typical aggregate split filenames.
    """
    all_files = glob.glob(os.path.join(input_dir, pattern))
    docs = []
    for p in all_files:
        base = os.path.basename(p)
        if base.lower() in AGGREGATE_BASENAMES:
            continue
        if os.path.isfile(p):
            docs.append(p)
    # Sort for deterministic order prior to shuffle
    docs.sort(key=lambda x: x.lower())
    return docs


def normalize_doc_text(text: str) -> str:
    """Normalize newlines to \n, strip trailing spaces per line, and ensure exactly one
    blank line at the end of the document so concatenation stays clean.
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces on each line
    lines = [ln.rstrip() for ln in text.split("\n")]
    # Ensure exactly one empty line at end
    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")  # one blank line to terminate the document
    return "\n".join(lines) + "\n"


def read_doc(path: str) -> str:
    with io.open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return normalize_doc_text(raw)


def count_tokens_and_sentences(doc_text: str) -> Tuple[int, int]:
    """Crude counts for diagnostics: token lines (non-empty) and sentence breaks (empty lines)."""
    token_lines = 0
    sentence_breaks = 0
    for ln in doc_text.split("\n"):
        if ln.strip() == "":
            sentence_breaks += 1
        else:
            token_lines += 1
    return token_lines, sentence_breaks


def largest_remainder_counts(n: int, ratios: List[float]) -> List[int]:
    """Allocate n items across k buckets using the Largest Remainder Method to honor ratios.
    Returns a list of integers that sum to n.
    """
    if n <= 0:
        return [0 for _ in ratios]
    # Normalize ratios if they don't sum to 1
    s = sum(ratios)
    if s <= 0:
        raise ValueError("Ratios must sum to a positive value.")
    ratios = [r / s for r in ratios]

    exact = [n * r for r in ratios]
    floors = [math.floor(x) for x in exact]
    assigned = sum(floors)
    remainders = [(x - f, i) for i, (x, f) in enumerate(zip(exact, floors))]
    remainders.sort(reverse=True)  # largest fractional parts first
    # Distribute remainder
    i = 0
    while assigned < n:
        _, idx = remainders[i % len(ratios)]
        floors[idx] += 1
        assigned += 1
        i += 1
    return floors


def split_indices(num_docs: int, train: float, val: float, test: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    """Shuffle indices and split by document using requested ratios."""
    idx = list(range(num_docs))
    random.Random(seed).shuffle(idx)

    counts = largest_remainder_counts(num_docs, [train, val, test])
    n_train, n_val, n_test = counts

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:n_train + n_val + n_test]

    return train_idx, val_idx, test_idx


def write_split(docs: List[str], indices: List[int], out_path: str) -> Dict[str, int]:
    """Concatenate the selected document texts into a single file out_path.
    Returns basic stats for diagnostics.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total_tokens = 0
    total_sent_breaks = 0
    with io.open(out_path, "w", encoding="utf-8", newline="\n") as w:
        for i in indices:
            content = read_doc(docs[i])
            tok, sb = count_tokens_and_sentences(content)
            total_tokens += tok
            total_sent_breaks += sb
            w.write(content)
    return {"documents": len(indices), "tokens": total_tokens, "sentence_breaks": total_sent_breaks}


def main():
    ap = argparse.ArgumentParser(description="Split multiple BIO2 .txt docs into train/val/test (by documents)")
    ap.add_argument("--input_dir", required=True, type=str, help="Folder with many BIO2 .txt files (one doc per file)")
    ap.add_argument("--output_dir", required=True, type=str, help="Folder to write train.txt / val.txt / test.txt")
    ap.add_argument("--train", type=float, default=0.7, help="Train ratio (default 0.7)")
    ap.add_argument("--val", type=float, default=0.1, help="Validation ratio (default 0.1)")
    ap.add_argument("--test", type=float, default=0.2, help="Test ratio (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling documents")
    ap.add_argument("--pattern", type=str, default="*.txt", help="Glob pattern for input files (default *.txt)")
    ap.add_argument("--write_dev_alias", action="store_true", help="Also write dev.txt as a copy of val.txt for trainer compatibility")
    args = ap.parse_args()

    # Collect docs
    docs = list_documents(args.input_dir, args.pattern)
    if not docs:
        raise SystemExit(f"No input .txt files found in: {args.input_dir}")

    # Report
    print(f"Found {len(docs)} candidate documents under {args.input_dir} (pattern={args.pattern}).")

    # Split indices
    train_idx, val_idx, test_idx = split_indices(len(docs), args.train, args.val, args.test, args.seed)

    # Write splits
    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "val.txt")
    test_path = os.path.join(args.output_dir, "test.txt")

    stats = {
        "train": write_split(docs, train_idx, train_path),
        "val": write_split(docs, val_idx, val_path),
        "test": write_split(docs, test_idx, test_path),
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "seed": args.seed,
    }

    # Optional dev.txt alias for compatibility with existing training scripts
    if args.write_dev_alias:
        dev_path = os.path.join(args.output_dir, "dev.txt")
        # Copy val.txt content
        with io.open(val_path, "r", encoding="utf-8") as rf, io.open(dev_path, "w", encoding="utf-8", newline="\n") as wf:
            wf.write(rf.read())
        stats["dev_alias"] = "dev.txt duplicated from val.txt"

    # Save manifest mapping files to splits
    manifest = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "train_files": [docs[i] for i in train_idx],
        "val_files": [docs[i] for i in val_idx],
        "test_files": [docs[i] for i in test_idx],
        "stats": stats,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with io.open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    # Pretty print summary
    def _fmt(s):
        return f"docs={s['documents']}, tokens={s['tokens']}, sentence_breaks={s['sentence_breaks']}"

    print("\nSplit summary:")
    print(f"  train: {_fmt(stats['train'])} -> {train_path}")
    print(f"  val  : {_fmt(stats['val'])} -> {val_path}")
    print(f"  test : {_fmt(stats['test'])} -> {test_path}")
    if args.write_dev_alias:
        print(f"  dev  : alias of val.txt -> {os.path.join(args.output_dir, 'dev.txt')}")


if __name__ == "__main__":
    main()

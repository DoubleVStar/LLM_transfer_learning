#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune a DeepSeek 8B (LLaMA-like) model for NER (BIO2) with QLoRA.

亮点：
- 子词对齐：offset_mapping（SentencePiece 友好）
- 训练稳定性：use_cache=False + gradient checkpointing
- LoRA 目标模块：自动探测（q/k/v/o_proj, wqkv/wo, gate/up/down_proj）
- 类别不平衡：自定义 Trainer，对 “O” 类降权（可调）
- 早停：EarlyStoppingCallback
- 新增：单进程模型并行（device_map=auto），无需 torchrun
- 新增：--hf_token / --local_files_only 支持私有模型与离线加载

依赖：
pip install --upgrade "transformers>=4.40" peft accelerate bitsandbytes seqeval datasets scikit-learn
"""

import os, io, json, argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# PEFT / QLoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Metrics
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report as seqeval_report
from torch.nn import CrossEntropyLoss


# -----------------------------
# Data reading (BIO2)
# -----------------------------

def read_bio2(path: str) -> List[Tuple[List[str], List[str]]]:
    """
    Read a single BIO2 file -> list of (tokens, tags) sentences.
    Line format: "token tag"   (space/tab split)
    Blank line separates sentences.
    """
    sents = []
    tokens, tags = [], []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    sents.append((tokens, tags))
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                token, tag = parts[0], "O"
            else:
                token, tag = " ".join(parts[:-1]), parts[-1]
            tokens.append(token)
            tags.append(tag)
    if tokens:
        sents.append((tokens, tags))
    return sents


def load_bio2_folder(folder: str) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """
    Expected files inside folder (any subset): train.txt, dev.txt/val.txt, test.txt
    """
    split_files = {}
    for name in ["train", "dev", "val", "test"]:
        for cand in [f"{name}.txt", f"{name}.bio", f"{name}.bio2"]:
            p = os.path.join(folder, cand)
            if os.path.exists(p):
                key = "val" if name == "dev" else name
                split_files[key] = p
                break
    data = {}
    for split, p in split_files.items():
        data[split] = read_bio2(p)
    if "train" not in data:
        raise FileNotFoundError("No train file found (expect train.txt / .bio / .bio2 in the folder).")
    return data


# -----------------------------
# Label utils & BIO fix
# -----------------------------

def collect_labels(train_sents: List[Tuple[List[str], List[str]]]) -> List[str]:
    labels = set()
    for _, ts in train_sents:
        for t in ts:
            labels.add(t)
    labels = sorted(labels)
    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels
    return labels


def fix_bio_sequence(tags: List[str]) -> List[str]:
    """Ensure BIO legality: if I-X follows O or different type, convert to B-X."""
    fixed = []
    prev_type = "O"
    for t in tags:
        if t == "O":
            fixed.append("O")
            prev_type = "O"
            continue
        if "-" not in t:
            fixed.append(t)
            prev_type = t
            continue
        bi, tp = t.split("-", 1)
        if bi == "B":
            fixed.append(t)
            prev_type = tp
        elif bi == "I":
            if prev_type == "O" or prev_type != tp:
                fixed.append("B-" + tp)
            else:
                fixed.append(t)
            prev_type = tp
        else:
            fixed.append(t)
            prev_type = t
    return fixed


# -----------------------------
# Tokenization alignment via offset_mapping
# -----------------------------

@dataclass
class NERExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class NERDataset(Dataset):
    def __init__(
        self,
        sents: List[Tuple[List[str], List[str]]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 512,
        label_all_tokens: bool = False,
        fix_bio: bool = True,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        self.fix_bio = fix_bio
        self.enc = self._encode(sents)

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        ex = self.enc[idx]
        return {
            "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(ex.attention_mask, dtype=torch.long),
            "labels": torch.tensor(ex.labels, dtype=torch.long),
        }

    def _encode(self, sents):
        out = []
        for tokens, tags in tqdm(sents, desc="Tokenizing"):
            if self.fix_bio:
                tags = fix_bio_sequence(tags)

            # 1) join tokens with single spaces, compute word spans
            text = " ".join(tokens)
            spans = []
            cursor = 0
            for w in tokens:
                start = cursor
                end = start + len(w)
                spans.append((start, end))
                cursor = end + 1  # +1 for the space

            # 2) tokenizer with offset_mapping (no padding here; collator will pad)
            enc = self.tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            offsets = enc["offset_mapping"]
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]

            # 3) map token offsets to word id -> labels
            labels = []
            wid_ptr = 0
            prev_wid = None

            for (st, ed) in offsets:
                if st == ed == 0:  # special tokens
                    labels.append(-100)
                    prev_wid = None
                    continue

                while wid_ptr < len(spans) and spans[wid_ptr][1] <= st:
                    wid_ptr += 1

                wid = None
                if wid_ptr < len(spans):
                    ws, we = spans[wid_ptr]
                    if st >= ws and ed <= we:
                        wid = wid_ptr

                if wid is None:
                    labels.append(-100)
                    prev_wid = None
                    continue

                tag = tags[wid]
                if wid != prev_wid:
                    labels.append(self.label2id[tag])  # first sub-token
                else:
                    if self.label_all_tokens:
                        if tag.startswith("B-"):
                            tag = "I-" + tag[2:]
                        labels.append(self.label2id.get(tag, self.label2id["O"]))
                    else:
                        labels.append(-100)
                prev_wid = wid

            out.append(NERExample(
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels
            ))
        return out


# -----------------------------
# Metrics (seqeval)
# -----------------------------

def build_compute_metrics(id2label: Dict[int, str]):
    def _to_tag_lists(batch_labels, batch_preds):
        y_true, y_pred = [], []
        for labels, preds in zip(batch_labels, batch_preds):
            lab_seq, pred_seq = [], []
            for l, p in zip(labels, preds):
                if l == -100:
                    continue
                lab_seq.append(id2label[int(l)])
                pred_seq.append(id2label[int(p)])
            y_true.append(lab_seq)
            y_pred.append(pred_seq)
        return y_true, y_pred

    def compute_metrics(p):
        logits = p.predictions
        preds = np.argmax(logits, axis=-1)
        labels = p.label_ids
        y_true, y_pred = _to_tag_lists(labels, preds)
        return {
            "precision": precision_score(y_true, y_pred),
            "recall":    recall_score(y_true, y_pred),
            "f1":        f1_score(y_true, y_pred),
        }
    return compute_metrics


# -----------------------------
# QLoRA helpers
# -----------------------------

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

def guess_lora_targets(model) -> List[str]:
    """Auto-detect LoRA target module names across different repos."""
    names = [n for n, _ in model.named_modules()]
    candidates = ["q_proj","k_proj","v_proj","o_proj","wqkv","wo","gate_proj","up_proj","down_proj"]
    targets = [c for c in candidates if any(n.endswith(c) or f".{c}" in n for n in names)]
    if not targets:
        targets = ["q_proj","k_proj","v_proj","o_proj"]
    print("[LoRA targets] ", sorted(set(targets)))
    return sorted(set(targets))


# -----------------------------
# Custom Trainer (O-class weighting)
# -----------------------------
# 放在 imports 区域加一个工具函数（可与 NERTrainer 放一起）
def unwrap_model(m):
    # 优先从 accelerate 提供的工具解包（可选）
    try:
        from accelerate.utils import extract_model_from_parallel
        m2 = extract_model_from_parallel(m)
    except Exception:
        m2 = getattr(m, "module", m)          # DDP
        m2 = getattr(m2, "base_model", m2)    # 一些 PEFT 包装
        m2 = getattr(m2, "model", m2)         # 另一些 PEFT 包装
    return m2

class NERTrainer(Trainer):
    def __init__(self, *args, o_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.o_weight = o_weight

    # 兼容新版 Trainer：接受 num_items_in_batch / **kwargs
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # 解包 DDP/PEFT，拿到真正的模型
        real_model = unwrap_model(model)
        labels = inputs.get("labels", None)
        if labels is None:
            # 兜底：交回给父类（避免未来接口变更导致问题）
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch, **kwargs
            )

        # 不就地修改 inputs，构造不含 labels 的输入
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits  # [B, L, C]
        C = logits.size(-1)

        # 类别权重（降低 'O'）：如果没找到 config/label2id 或 o_weight<=0 就退化为等权
        weight = torch.ones(C, device=logits.device)
        try:
            l2i = getattr(getattr(real_model, "config", None), "label2id", None)
            if self.o_weight > 0 and isinstance(l2i, dict) and "O" in l2i:
                weight[l2i["O"]] = self.o_weight
        except Exception:
            pass

        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, C), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser("DeepSeek 8B QLoRA NER (BIO2) — single process friendly")

    # Data & model
    ap.add_argument("--data_dir", required=True, help="Folder with train.txt / dev|val.txt / test.txt (BIO2)")
    ap.add_argument("--base_model", required=True, help="HF repo id or local path of the base model")
    ap.add_argument("--output_dir", required=True)

    # Tokenizer/model loading auth & mode
    ap.add_argument("--hf_token", type=str, default=None,
                    help="HF token for private/gated repos; or set env HF_TOKEN/HUGGINGFACE_HUB_TOKEN")
    ap.add_argument("--local_files_only", action="store_true",
                    help="Only use local cached files (offline)")

    # Single-process model parallel (no torchrun)
    ap.add_argument("--single_process_mp", action="store_true",
                    help="Use single-process model parallel via device_map=auto (no torchrun).")
    ap.add_argument("--gpu_mem", type=str, default="22GiB",
                    help="Per-GPU max memory for device_map=auto, e.g. '22GiB' for 24GB cards.")

    # Train knobs
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--label_all_tokens", action="store_true")
    ap.add_argument("--fix_bio", action="store_true", help="Fix illegal BIO sequences (I-X -> B-X when needed)")
    ap.add_argument("--seed", type=int, default=42)

    # Logging / eval / precision
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--bf16", action="store_true", help="force bf16, else auto-detect by device")
    ap.add_argument("--fp16", action="store_true", help="fallback to fp16 if needed")
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass through to AutoModel/Tokenizer")

    # Loss & early stopping
    ap.add_argument("--o_weight", type=float, default=0.1, help="class weight for 'O' in CE loss (0 disables)")
    ap.add_argument("--es_patience", type=int, default=2, help="early stopping patience on dev F1")

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # -- auth kwargs for private/gated repos
    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    auth = {}
    if token:
        # transformers 新旧版本兼容
        try:
            auth = {"token": token}
        except TypeError:
            auth = {"use_auth_token": token}

    # 1) Load BIO2 data
    data = load_bio2_folder(args.data_dir)
    train_sents = data["train"]
    val_sents = data.get("val", [])
    test_sents = data.get("test", [])

    # 2) Build labels
    labels = collect_labels(train_sents)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    with io.open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "label2id": label2id, "id2label": id2label}, f, indent=2, ensure_ascii=False)

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        **auth
    )
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass

    # 4) Model (QLoRA)
    bnb_config = get_bnb_config()
    config = AutoConfig.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        **auth
    )

    device_map_kw = {}
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    if args.single_process_mp and torch.cuda.is_available():
        # 单进程模型并行（不用 torchrun）
        n_gpu = torch.cuda.device_count()
        max_memory = {i: args.gpu_mem for i in range(n_gpu)}
        device_map_kw = dict(device_map="auto", max_memory=max_memory)
        print(f"[MP] Single-process model parallel enabled on {n_gpu} GPUs with max_memory={max_memory}")
    elif ddp_local_rank != -1:
        # DDP: 每个 rank 直接把 4bit 模型加载到自己的 GPU
        torch.cuda.set_device(ddp_local_rank)
        device_map_kw = dict(device_map={"": ddp_local_rank})
        print(f"[DDP] rank={ddp_local_rank}: loading 4-bit model on cuda:{ddp_local_rank}")
    else:
        # 单 GPU、非 DDP
        print("[MP] Single-process MP disabled; loading model on default device.")


    base = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        config=config,
        quantization_config=bnb_config,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        **device_map_kw,
        **auth
    )

    # 重要：训练稳定性
    if hasattr(base, "config"):
        base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        try:
            # 需要 torch>=2.1（建议2.1+），transformers>=4.34+
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # 你的版本不支持时，先降级为默认实现；若仍报错，再考虑临时关闭GC
            base.gradient_checkpointing_enable()


    # prepare k-bit + LoRA
    base = prepare_model_for_kbit_training(base)
    targets = guess_lora_targets(base)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none",
        target_modules=targets,
        task_type="TOKEN_CLS",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    # Single-process MP hint to HF Trainer
    if args.single_process_mp:
        model.is_model_parallel = True
        model.model_parallel = True

    # 5) Datasets（动态 padding 交给 collator）
    train_ds = NERDataset(train_sents, tokenizer, label2id, args.max_length, args.label_all_tokens, fix_bio=args.fix_bio)
    val_ds = NERDataset(val_sents, tokenizer, label2id, args.max_length, args.label_all_tokens, fix_bio=args.fix_bio) if val_sents else None
    test_ds = NERDataset(test_sents, tokenizer, label2id, args.max_length, args.label_all_tokens, fix_bio=args.fix_bio) if test_sents else None

    # 6) Collator
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8)

    # 7) TrainingArguments
    common = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.grad_accum,
        save_total_limit=2,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=2,
        fp16=args.fp16,
        bf16=args.bf16 or (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        label_names=["labels"],
        ddp_find_unused_parameters=False,
    )
    try:
        training_args = TrainingArguments(eval_strategy="steps", eval_steps=args.eval_steps, **common)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="steps", eval_steps=args.eval_steps, **common)

    # 8) Trainer（O 类降权 + 早停）
    callbacks = []
    if args.es_patience and args.es_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.es_patience, early_stopping_threshold=1e-4))

    trainer = NERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(id2label),
        callbacks=callbacks,
        o_weight=args.o_weight,
    )

    # 9) Train
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with io.open(os.path.join(args.output_dir, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)

    # 10) Evaluate (val + optional test). Also dump seqeval report (digits=4)
    def eval_and_dump(name, ds):
        if ds is None or len(ds) == 0:
            return
        metrics = trainer.evaluate(eval_dataset=ds)
        with io.open(os.path.join(args.output_dir, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        preds = trainer.predict(ds)
        logits = preds.predictions
        pred_ids = np.argmax(logits, axis=-1)
        label_ids = preds.label_ids
        y_true_list, y_pred_list = [], []
        for labs, prds in zip(label_ids, pred_ids):
            t_seq, p_seq = [], []
            for l, p in zip(labs, prds):
                if l == -100:
                    continue
                t_seq.append(id2label[int(l)])
                p_seq.append(id2label[int(p)])
            y_true_list.append(t_seq)
            y_pred_list.append(p_seq)
        rep = seqeval_report(y_true_list, y_pred_list, digits=4)
        with io.open(os.path.join(args.output_dir, f"{name}_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep)

    eval_and_dump("val", val_ds)
    eval_and_dump("test", test_ds)

    print("\n[Done] DeepSeek NER QLoRA fine-tuning finished.")
    print(f"Model & adapters saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

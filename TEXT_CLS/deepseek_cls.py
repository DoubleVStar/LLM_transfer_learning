


import os, io, json, argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from transformers import BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 尝试 safetensors 读取
try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


# -----------------------------
# Helpers
# -----------------------------
def read_csv_any(path: str) -> pd.DataFrame:
    encs = [None, "utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last = None
    for e in encs:
        try:
            return pd.read_csv(path, encoding=e)
        except Exception as ex:
            last = ex
    raise last

def clean_str(x):
    s = str(x).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s

def unwrap_model(m):
    try:
        from accelerate.utils import extract_model_from_parallel
        return extract_model_from_parallel(m)
    except Exception:
        m2 = getattr(m, "module", m)       # DDP
        m2 = getattr(m2, "base_model", m2) # PEFT 包装之一
        m2 = getattr(m2, "model", m2)      # PEFT 包装之二
        return m2

def guess_lora_targets(model) -> List[str]:
    names = [n for n, _ in model.named_modules()]
    candidates = ["q_proj","k_proj","v_proj","o_proj","wqkv","wo","gate_proj","up_proj","down_proj"]
    targets = [c for c in candidates if any(n.endswith(c) or f".{c}" in n for n in names)]
    if not targets:
        targets = ["q_proj","k_proj","v_proj","o_proj"]
    print("[LoRA targets] ", sorted(set(targets)))
    return sorted(set(targets))

def load_adapter_config(adapter_dir: str) -> dict:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with io.open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_lora_from_adapter_config(cfg: dict, fallback_targets: List[str]) -> LoraConfig:
    r = int(cfg.get("r", 16))
    lora_alpha = int(cfg.get("lora_alpha", 32))
    lora_dropout = float(cfg.get("lora_dropout", 0.05))
    bias = cfg.get("bias", "none")
    targets = cfg.get("target_modules", None) or fallback_targets
    # 无论原任务是 TOKEN_CLS 还是别的，这里目标任务是 SEQ_CLS
    return LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias,
        target_modules=targets, task_type="SEQ_CLS",
    )

def load_lora_state_dict_only(adapter_dir: str) -> Dict[str, torch.Tensor]:
    """
    只提取 LoRA 权重；过滤掉 modules_to_save（例如 score 头）等非 LoRA 项。
    """
    # 可能的权重文件名
    for fn in ["adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin"]:
        p = os.path.join(adapter_dir, fn)
        if os.path.exists(p):
            if fn.endswith(".safetensors") and safe_load_file is not None:
                sd = safe_load_file(p, device="cpu")
            else:
                sd = torch.load(p, map_location="cpu")
            break
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    filtered = {}
    removed = []
    for k, v in sd.items():
        # 典型 LoRA 权重包含 lora_A / lora_B / lora_embedding_A/B
        is_lora = ("lora_" in k) or ("lora_embedding_" in k)
        # modules_to_save 通常用于保存任务头（score/cls 等），需要过滤
        is_modules_to_save = ("modules_to_save" in k) or (".score." in k)
        if is_lora and (not is_modules_to_save):
            filtered[k] = v
        else:
            removed.append(k)

    print(f"[INIT] Filtered adapter keys: kept {len(filtered)} LoRA tensors; removed {len(removed)} non-LoRA keys.")
    return filtered


# -----------------------------
# Dataset
# -----------------------------
class CLSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str,
                 label2id: Dict[str,int], max_length: int = 512):
        self.tokenizer = tokenizer
        self.texts = [clean_str(t) for t in df[text_col].tolist()]
        self.labels = [label2id[str(l)] for l in df[label_col].astype(str).tolist()]
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        enc["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        # ensure tensors
        for k, v in list(enc.items()):
            if isinstance(v, list):
                enc[k] = torch.tensor(v)
        return enc


# -----------------------------
# Metrics
# -----------------------------
def build_compute_metrics(id2label: Dict[int,str]):
    def compute_metrics(eval_pred):
        preds_arr = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
        labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
        if isinstance(preds_arr, tuple):
            preds_arr = preds_arr[0]
        preds = np.argmax(preds_arr, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
        return {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro, "f1_weighted": f1_weighted}
    return compute_metrics


# -----------------------------
# Custom Trainer (optional class weights)
# -----------------------------
class CLSTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels", None)
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch, **kwargs)
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits  # [B, C]
        C = logits.size(-1)
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=w)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, C), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser("DeepSeek 8B QLoRA — TEXT classification")

    # Data
    ap.add_argument("--train_csv", default='/home/kevin/MultiTagger-v2/DATA/TEXT_CLS/data_split/train.csv')
    ap.add_argument("--dev_csv", default='/home/kevin/MultiTagger-v2/DATA/TEXT_CLS/data_split/dev.csv')
    ap.add_argument("--test_csv", default='/home/kevin/MultiTagger-v2/DATA/TEXT_CLS/data_split/test.csv')
    ap.add_argument("--text_col", default="TEXT")
    ap.add_argument("--label_col", default="LABEL")
    ap.add_argument("--val_ratio", type=float, default=0.0, help="If dev_csv not given, split from train (e.g., 0.1)")
    ap.add_argument("--test_ratio", type=float, default=0.0, help="If test_csv not given, split from train (e.g., 0.1)")
    ap.add_argument("--drop_unknown", action="store_true", help="Drop rows in dev/test with labels unseen in train")

    # Model IO
    ap.add_argument("--base_model", required=True, help="HF repo id or local path")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--hf_token", type=str, default=None, help="HF token; or set env HF_TOKEN/HUGGINGFACE_HUB_TOKEN")
    ap.add_argument("--local_files_only", action="store_true")

    # Use NER adapter as initialization
    ap.add_argument("--init_adapter", type=str, default=None,
                    help="Path to a PEFT/LoRA adapter (e.g., your NER run output). LoRA-only weights will be loaded.")

    # Parallel modes
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")

    # Eval/logging
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--es_patience", type=int, default=2)

    # Imbalance handling
    ap.add_argument("--class_weight", choices=["none","auto"], default="none",
                    help="'auto' uses inverse frequency from train set")

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Auth for private/gated repos
    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    auth = {}
    if token:
        try: auth = {"token": token}
        except TypeError: auth = {"use_auth_token": token}

    # ---- Load data
    df_tr = read_csv_any(args.train_csv)
    df_va = read_csv_any(args.dev_csv) if args.dev_csv is not None else None
    df_te = read_csv_any(args.test_csv) if args.test_csv is not None else None

    # column checks
    for df_name, df in [("train", df_tr), ("dev", df_va), ("test", df_te)]:
        if df is None: continue
        if args.text_col not in df.columns or args.label_col not in df.columns:
            raise ValueError(f"{df_name} missing required columns: {args.text_col} | {args.label_col}")

    # split from train if needed
    if df_va is None or df_te is None:
        from sklearn.model_selection import train_test_split
        rest = df_tr
        test_ratio = args.test_ratio if df_te is None else 0.0
        val_ratio = args.val_ratio if df_va is None else 0.0
        tot = test_ratio + val_ratio
        if tot > 0:
            train_part, temp = train_test_split(rest, test_size=tot, random_state=args.seed, stratify=rest[args.label_col])
            if val_ratio > 0 and test_ratio > 0:
                va_size = val_ratio / tot
                df_va, df_te = train_test_split(temp, test_size=(1 - va_size), random_state=args.seed,
                                                stratify=temp[args.label_col])
            elif val_ratio > 0:
                df_va, df_te = temp, df_te
            elif test_ratio > 0:
                df_va, df_te = df_va, temp
            df_tr = train_part

    # cleaning
    for df in [df_tr, df_va, df_te]:
        if df is None: continue
        df[args.text_col] = df[args.text_col].map(clean_str)
        df[args.label_col] = df[args.label_col].map(clean_str)
        df.dropna(subset=[args.text_col, args.label_col], inplace=True)

    # label mapping (from TRAIN)
    labels_sorted = sorted(df_tr[args.label_col].astype(str).unique().tolist())
    label2id = {l:i for i,l in enumerate(labels_sorted)}
    id2label = {i:l for l,i in label2id.items()}

    # drop_unknown on dev/test
    if args.drop_unknown:
        if df_va is not None:
            df_va = df_va[df_va[args.label_col].astype(str).isin(label2id.keys())].copy()
        if df_te is not None:
            df_te = df_te[df_te[args.label_col].astype(str).isin(label2id.keys())].copy()

    # class weights (from train)
    class_weights_tensor = None
    if args.class_weight == "auto":
        counts = df_tr[args.label_col].value_counts()
        freq = counts.reindex(labels_sorted).fillna(0).astype(float).values
        inv = 1.0 / np.clip(freq, 1.0, None)
        w = inv / inv.sum() * len(inv)  # normalize to mean 1
        class_weights_tensor = torch.tensor(w, dtype=torch.float32)
        print("[class_weight:auto]", dict(zip(labels_sorted, w.round(6))))

    # ---- Tokenizer
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

    # PAD handling: use eos as pad if missing
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"[PAD] tokenizer.pad_token set to eos (id={tokenizer.pad_token_id})")
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            print(f"[PAD] tokenizer.pad_token added <pad> (id={tokenizer.pad_token_id})")
        tokenizer.padding_side = "right"

    # ---- 4-bit quant config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    # ---- Config
    config = AutoConfig.from_pretrained(
        args.base_model,
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        **auth
    )
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id

    # ---- Device mapping (DDP vs single process MP)
    device_map_kw = {}
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if args.single_process_mp and torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        max_memory = {i: args.gpu_mem for i in range(n_gpu)}
        device_map_kw = dict(device_map="auto", max_memory=max_memory)
        print(f"[MP] Single-process model parallel on {n_gpu} GPUs, max_memory={max_memory}")
    elif ddp_local_rank != -1:
        torch.cuda.set_device(ddp_local_rank)
        device_map_kw = dict(device_map={"": ddp_local_rank})
        print(f"[DDP] rank={ddp_local_rank}: loading 4-bit model on cuda:{ddp_local_rank}")
    else:
        print("[MP] Single-process MP disabled; loading on default device.")

    # ---- Base model
    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
        quantization_config=bnb_config,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        **device_map_kw,
        **auth
    )
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tokenizer.pad_token_id

    # 若 tokenizer 新增了 <pad>，需要 resize
    need_resize = (tokenizer.pad_token == "<pad>") or (
        hasattr(base, "get_input_embeddings")
        and tokenizer.pad_token_id is not None
        and len(tokenizer) != base.get_input_embeddings().num_embeddings
    )
    if need_resize:
        try:
            base.resize_token_embeddings(len(tokenizer))
            print(f"[PAD] resized token embeddings to {len(tokenizer)}")
        except Exception as e:
            print("[WARN] resize_token_embeddings failed:", e)

    # 稳定性
    if hasattr(base, "config"):
        base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        try:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            base.gradient_checkpointing_enable()

    # ---- QLoRA / 适配器初始化
    if args.init_adapter:
        print(f"[INIT] Using adapter from: {args.init_adapter}")
        base = prepare_model_for_kbit_training(base)
        # 读取适配器配置并据此构建 LoRA（目标模块和超参与 NER 一致）
        adapter_cfg = load_adapter_config(args.init_adapter)
        targets = guess_lora_targets(base)
        lora_cfg = build_lora_from_adapter_config(adapter_cfg, fallback_targets=targets)
        model = get_peft_model(base, lora_cfg)
        # 只加载 LoRA 权重，过滤掉 modules_to_save/score
        lora_sd = load_lora_state_dict_only(args.init_adapter)
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)
        print(f"[INIT] Loaded LoRA-only state_dict. missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        base = prepare_model_for_kbit_training(base)
        targets = guess_lora_targets(base)
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=targets, task_type="SEQ_CLS",
        )
        model = get_peft_model(base, lora_cfg)

    model.print_trainable_parameters()
    if args.single_process_mp:
        model.is_model_parallel = True
        model.model_parallel = True

    # ---- Datasets & collator
    train_ds = CLSDataset(df_tr, tokenizer, args.text_col, args.label_col, label2id, args.max_length)
    val_ds = CLSDataset(df_va, tokenizer, args.text_col, args.label_col, label2id, args.max_length) if df_va is not None else None
    test_ds = CLSDataset(df_te, tokenizer, args.text_col, args.label_col, label2id, args.max_length) if df_te is not None else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # ---- TrainingArguments
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
        metric_for_best_model="f1_macro",
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

    # ---- Trainer
    callbacks = []
    if args.es_patience and args.es_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.es_patience, early_stopping_threshold=1e-4))

    compute_metrics = build_compute_metrics(id2label)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    try:
        trainer = CLSTrainer(processing_class=tokenizer, class_weights=class_weights_tensor, **trainer_kwargs)
    except TypeError:
        trainer = CLSTrainer(tokenizer=tokenizer, class_weights=class_weights_tensor, **trainer_kwargs)

    # ---- Train
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with io.open(os.path.join(args.output_dir, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2, ensure_ascii=False)

    # ---- Evaluate helper (save classification_report with 4 decimals)
    def eval_and_dump(name, ds):
        if ds is None or len(ds) == 0: return
        metrics = trainer.evaluate(eval_dataset=ds)
        with io.open(os.path.join(args.output_dir, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        preds = trainer.predict(ds)
        y_true = preds.label_ids
        y_pred = np.argmax(preds.predictions, axis=-1)
        target_names = [id2label[i] for i in range(len(id2label))]
        rep = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
        with io.open(os.path.join(args.output_dir, f"{name}_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep)

    eval_and_dump("val", val_ds)
    eval_and_dump("test", test_ds)

    print("\n[Done] DeepSeek TEXT classification fine-tuning finished.")
    print(f"Model & adapters saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

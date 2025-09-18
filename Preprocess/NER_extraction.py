import os, sys, json, re, glob, argparse, csv, math
from collections import defaultdict
from typing import List, Tuple, Dict, Any

def split_sentences(text: str) -> List[str]:
    """Robust-ish sentence split: try NLTK if available; else regex fallback."""
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in parts if s and s.strip()]

def extract_iao_pairs(infons: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return list of (iao_id, iao_name) pairs found on a passage (may be multiple)."""
    idxs = set()
    for k in infons.keys():
        m = re.fullmatch(r"iao_id_(\d+)", k)
        if m:
            idxs.add(int(m.group(1)))
    for k in infons.keys():
        m = re.fullmatch(r"iao_name_(\d+)", k)
        if m:
            idxs.add(int(m.group(1)))
    pairs = []
    for i in sorted(idxs):
        iid = (infons.get(f"iao_id_{i}") or "").strip()  # e.g., "IAO:0000317"
        inm = (infons.get(f"iao_name_{i}") or "").strip()
        if iid or inm:
            pairs.append((iid, inm))
    return pairs

def iao_ids_only(pairs: List[Tuple[str,str]]) -> List[str]:
    """Convert ('IAO:0000317','...')-> '0000317'; tolerate missing prefix."""
    ids = []
    for iid, _ in pairs:
        if not iid:
            continue
        m = re.search(r'(\d{7})$', iid)
        ids.append(m.group(1) if m else iid)
    return ids

def load_pipeline(model_dir: str, max_length: int = 512, device: int = None):
    from transformers import AutoTokenizer, pipeline
    import torch
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, model_max_length=max_length)
    nlp = pipeline(
        task="token-classification",
        model=model_dir,
        tokenizer=tok,
        aggregation_strategy="simple",
        device=device,
    )
    return nlp

def ner_batch(nlp, sents: List[str], batch_size: int = 16) -> List[list]:
    if not sents:
        return []
    try:
        preds = nlp(sents, batch_size=batch_size)
    except TypeError:
        preds = nlp(sents)  # old pipelines
    out = []
    for ents in preds:
        norm = []
        for e in ents:
            eg = e.get("entity_group") or e.get("entity")
            norm.append({
                "entity_group": eg,
                "word": e.get("word"),
                "start": int(e.get("start", 0)) if e.get("start") is not None else None,
                "end": int(e.get("end", 0)) if e.get("end") is not None else None,
                "score": float(e.get("score", 0)) if e.get("score") is not None else None,
            })
        out.append(norm)
    return out

def format_iao_section(pairs: List[Tuple[str,str]], allowed_set: set) -> str:
    """Return a human-readable IAO section string for matched pairs within allowed_set ids."""
    parts = []
    for iid, inm in pairs:
        m = re.search(r'(\d{7})$', iid or '')
        last = m.group(1) if m else ''
        if last and last in allowed_set:
            label = f"{iid}" if iid else ""
            if inm:
                label = f"{label} {inm}".strip()
            parts.append(label)
    return "; ".join(parts) if parts else ""

def should_keep(pairs: List[Tuple[str,str]], allowed: set, mode: str) -> bool:
    ids = set(iao_ids_only(pairs))
    ids = {i for i in ids if i}
    if not ids:
        return False
    if mode == "subset":
        return ids.issubset(allowed) and len(ids) > 0
    return len(ids & allowed) > 0

def main():
    ap = argparse.ArgumentParser(description="Extract NER per sentence for passages whose IAO ids are in an allowed set.")
    ap.add_argument("--inputs", required=True, help="Glob or directory of PMC*_v1.json files")
    ap.add_argument("--model-dir", required=True, help="Fine-tuned token-classification model directory")
    ap.add_argument("--out-csv", required=True, help="Output CSV path (large DataFrame-like)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", type=int, default=0, help="GPU id (e.g., 0) or -1 for CPU; default auto")
    ap.add_argument("--filter-mode", choices=["subset","any"], default="subset",
                    help="subset: passage IAO ids must be a subset of allowed ids; any: overlap is enough")
    ap.add_argument("--keep-empty", action="store_true", help="Keep sentences without any NER entities")
    ap.add_argument("--allowed-iao", default="0000317,0000319,0000633,0000318,0000315,0000314,0000615,0000305",
                    help="Comma-separated IAO suffix ids to keep, e.g., '0000317,0000319'")
    args = ap.parse_args()

    if os.path.isdir(args.inputs):
        paths = sorted(glob.glob(os.path.join(args.inputs, "*.json")))
    else:
        paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise SystemExit(f"No files matched: {args.inputs}")

    allowed = set([x.strip() for x in args.allowed_iao.split(",") if x.strip()])
    nlp = load_pipeline(args.model_dir, max_length=args.max_length, device=args.device)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fout = open(args.out_csv, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=["PMCID", "ORIGINAL SENTENCE", "NER ANNOTATION", "IAO SECTION"])
    writer.writeheader()

    total_files = 0
    kept_passages = 0
    total_sent_rows = 0
    total_entities = 0

    for p in paths:
        total_files += 1
        base = os.path.basename(p)
        pmcid = os.path.splitext(base)[0]
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue

        docs = data.get("documents") or []
        for doc in docs:
            sent_blocks = []
            iao_blocks = []
            for pas in doc.get("passages", []):
                inf = pas.get("infons", {}) or {}
                pairs = extract_iao_pairs(inf)
                if not should_keep(pairs, allowed, args.filter_mode):
                    continue
                text = (pas.get("text") or "").strip()
                if not text:
                    continue
                sents = split_sentences(text)
                if not sents:
                    continue
                sent_blocks.append(sents)
                iao_blocks.append(pairs)

            if not sent_blocks:
                continue

            flat_sents = []
            mapping = []
            for bi, sents in enumerate(sent_blocks):
                for s in sents:
                    flat_sents.append(s)
                    mapping.append(bi)

            batched = ner_batch(nlp, flat_sents, batch_size=args.batch_size)

            for s, bi, ents in zip(flat_sents, mapping, batched):
                if not ents and not args.keep_empty:
                    continue
                iao_str = format_iao_section(iao_blocks[bi], allowed)
                writer.writerow({
                    "PMCID": pmcid,
                    "ORIGINAL SENTENCE": s,
                    "NER ANNOTATION": json.dumps(ents, ensure_ascii=False),
                    "IAO SECTION": iao_str
                })
                total_sent_rows += 1
                total_entities += len(ents)
            kept_passages += len(sent_blocks)

    fout.close()
    print(f"[INFO] Files processed: {total_files}")
    print(f"[INFO] Passages kept (IAO-filtered): {kept_passages}")
    print(f"[INFO] Sentence rows written: {total_sent_rows}")
    print(f"[INFO] Total entities extracted: {total_entities}")
    print(f"[OK] Saved to {args.out_csv}")

if __name__ == "__main__":
    main()
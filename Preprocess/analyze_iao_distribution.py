
import os, glob, json, re, argparse, csv
from collections import Counter, defaultdict

def extract_iao_pairs(infons):
    """Return a list of (iao_id, iao_name) pairs found in a passage's infons (can be multiple)."""
    idxs = set()
    for k in infons.keys():
        m = re.fullmatch(r"iao_id_(\d+)", k)
        if m: idxs.add(int(m.group(1)))
    for k in infons.keys():
        m = re.fullmatch(r"iao_name_(\d+)", k)
        if m: idxs.add(int(m.group(1)))
    pairs = []
    for i in sorted(idxs):
        iao_id = (infons.get(f"iao_id_{i}") or "").strip()
        iao_nm = (infons.get(f"iao_name_{i}") or "").strip()
        if iao_id or iao_nm:
            pairs.append((iao_id, iao_nm))
    return pairs

def load_pipeline(model_dir, max_length=512, device=None):
    # Lazy-import HF only when needed
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

def run_pipeline_batch(nlp, texts, max_length=512, batch_size=16):
    """Run pipeline on a list of texts with truncation; returns list[list[entity_dict]]."""
    if not texts:
        return []
    preds = nlp(texts, batch_size=batch_size)
    # Normalize keys
    norm_all = []
    for ents in preds:
        out = []
        for e in ents:
            eg = e.get("entity_group") or e.get("entity")
            out.append({
                "entity_group": eg,
                "word": e.get("word"),
                "start": int(e.get("start", 0)) if e.get("start") is not None else None,
                "end": int(e.get("end", 0)) if e.get("end") is not None else None,
            })
        norm_all.append(out)
    return norm_all

def analyze_files_with_model(paths, model_dir, assignment="duplicate", max_length=512, batch_size=16, device=None):
    """
    For each passage: run NER, count number of entities, and assign that count to ALL IAO pairs on the passage.
    If assignment='share', divide the count equally among pairs.
    Returns (overall_ctr, by_file_ctr, n_files, n_passages, total_entities).
    """
    nlp = load_pipeline(model_dir, max_length=max_length, device=device)

    overall = Counter()
    by_file = defaultdict(Counter)
    n_files = 0
    n_passages = 0
    total_entities = 0

    for p in paths:
        n_files += 1
        base = os.path.basename(p)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue

        docs = data.get("documents") or []
        for doc in docs:
            # Collect all passages of this doc for a single batch inference
            texts, iao_pairs_list = [], []
            for pas in doc.get("passages", []):
                inf = pas.get("infons", {}) or {}
                text = (pas.get("text") or "").strip()
                pairs = extract_iao_pairs(inf)
                if not text:
                    # still count 0 into iao pairs? Typically no, skip empty text
                    continue
                texts.append(text)
                iao_pairs_list.append(pairs)

            # Run model in batch
            batched_ents = run_pipeline_batch(nlp, texts, max_length=max_length, batch_size=batch_size)

            # Aggregate counts
            for pairs, ents in zip(iao_pairs_list, batched_ents):
                n_passages += 1
                k = len(pairs)
                c = float(len(ents))
                total_entities += len(ents)
                if k == 0:
                    # No IAO pairs on this passage: ignore for IAO stats
                    continue
                if assignment == "share" and k > 0:
                    per = c / k
                    for pair in pairs:
                        overall[pair] += per
                        by_file[base][pair] += per
                else:  # duplicate
                    for pair in pairs:
                        overall[pair] += c
                        by_file[base][pair] += c

    return overall, by_file, n_files, n_passages, total_entities

def save_csvs(overall, by_file, outdir):
    os.makedirs(outdir, exist_ok=True)
    dist_csv = os.path.join(outdir, "iao_model_distribution.csv")
    with open(dist_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iao_id","iao_name","count"])
        for (iid, inm), c in sorted(overall.items(), key=lambda x: (-x[1], x[0])):
            # write float counts with up to 6 decimals if sharing
            w.writerow([iid, inm, f"{c:.6f}" if isinstance(c, float) and not c.is_integer() else int(c)])

    dist_by_file_csv = os.path.join(outdir, "iao_model_distribution_by_file.csv")
    with open(dist_by_file_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file","iao_id","iao_name","count"])
        for fname, ctr in sorted(by_file.items()):
            for (iid, inm), c in sorted(ctr.items(), key=lambda x: (-x[1], x[0])):
                w.writerow([fname, iid, inm, f"{c:.6f}" if isinstance(c, float) and not c.is_integer() else int(c)])
    return dist_csv, dist_by_file_csv

def plot_overall(overall, outdir, top_n=None):
    import matplotlib.pyplot as plt
    items = sorted(overall.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        items = items[:top_n]
    labels = [f"{iid or ''}\n{inm or ''}".strip() for (iid,inm),_ in items]
    values = [c for _, c in items]

    width = 12
    height = 5 if len(values) <= 20 else min(16, 5 + (len(values)-20)*0.25)

    plt.figure(figsize=(width, height))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=55, ha="right")
    plt.ylabel("NER entity count (model)")
    plt.title("IAO Distribution by Model-Detected Entities (All Files)")
    plt.tight_layout()
    out_png = os.path.join(outdir, "iao_model_distribution.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png

def main():
    ap = argparse.ArgumentParser(description="Use a trained NER model to extract entities and aggregate counts by IAO across many JSON files.")
    ap.add_argument("--inputs", required=True, help="Directory or glob for JSON files, e.g., '/path/*.json' or '/path/to/dir'")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--model-dir", required=True, help="Path to fine-tuned token-classification model directory")
    ap.add_argument("--batch-size", type=int, default=16, help="Pipeline batch size")
    ap.add_argument("--max-length", type=int, default=512, help="Tokenizer/pipeline max token length")
    ap.add_argument("--device", type=int, default=None, help="GPU id (e.g., 0) or -1 for CPU; default auto")
    ap.add_argument("--assignment", choices=["duplicate","share"], default="duplicate",
                    help="If a passage has multiple IAO pairs, duplicate entity counts to each IAO (duplicate), or share equally (share).")
    ap.add_argument("--top-n", type=int, default=None, help="Only plot top-N IAO entries")
    ap.add_argument("--top-frac", type=float, default=None, help="Plot top fraction")



    args = ap.parse_args()

    if os.path.isdir(args.inputs):
        paths = sorted(glob.glob(os.path.join(args.inputs, "*.json")))
    else:
        paths = sorted(glob.glob(args.inputs))

    if not paths:
        raise SystemExit(f"No files matched: {args.inputs}")

    overall, by_file, n_files, n_passages, total_entities = analyze_files_with_model(
        paths,
        model_dir=args.model_dir,
        assignment=args.assignment,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device
    )

    print(f"[INFO] Files: {n_files}, Passages (with text): {n_passages}, Total model entities: {total_entities}, Unique IAO pairs: {len(overall)}")

    dist_csv, dist_by_file_csv = save_csvs(overall, by_file, args.outdir)

    if args.top_frac:
        top_n = max(1, int(len(overall) * args.top_frac))
    else:
        top_n = args.top_n
    out_png = plot_overall(overall, args.outdir, top_n=top_n)
    print("Saved:", dist_csv)
    print("Saved:", dist_by_file_csv)
    print("Saved:", out_png)

if __name__ == "__main__":
    main()

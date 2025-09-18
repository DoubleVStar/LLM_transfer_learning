# ner_eval.py
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

# =========================
# 1) Tag parsing / span decoding
# =========================

def _split_tag(tag: str) -> Tuple[str, Optional[str]]:
    """Split tag into prefix and type. 'O' -> ('O', None). Supports BIO/BIOES/BILOU."""
    if tag in (None, "", "O"):
        return "O", None
    if "-" not in tag:
        return "O", None
    pfx, typ = tag.split("-", 1)
    return pfx.upper(), typ

def tags_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Decode a tag sequence into entity spans.
    Returns a list of (start, end, type), with end being exclusive (span = tokens[start:end]).
    Supports BIO/IOB2 and BIOES/BILOU. Includes basic robustness for malformed sequences.
    """
    spans: List[Tuple[int, int, str]] = []
    start, ent_type = None, None

    def close_entity(end_idx: int):
        nonlocal start, ent_type
        if start is not None and ent_type is not None and end_idx > start:
            spans.append((start, end_idx, ent_type))
        start, ent_type = None, None

    for i, tag in enumerate(list(tags) + ["O"]):  # sentinel to flush the last open span
        pfx, typ = _split_tag(tag)
        if pfx in {"B", "S", "U"}:
            close_entity(i)
            start, ent_type = i, typ
            if pfx in {"S", "U"}:  # single-token entity
                close_entity(i + 1)
        elif pfx == "I":
            if start is None or ent_type != typ:  # tolerance: treat broken I-* as a new B-*
                close_entity(i)
                start, ent_type = i, typ
        elif pfx in {"L", "E"}:
            if start is None or ent_type != typ:  # tolerance: treat lone L/E as single-token
                start, ent_type = i, typ
            close_entity(i + 1)
        else:  # 'O' or invalid
            close_entity(i)
    return spans

# =========================
# 2) Metric helpers
# =========================

def _span_len(span: Tuple[int, int]) -> int:
    return max(0, span[1] - span[0])

def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def _iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = _overlap(a, b)
    union = _span_len(a) + _span_len(b) - inter
    return (inter / union) if union > 0 else 0.0

def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}

# =========================
# 3) Entity-level metrics (strict / partial)
# =========================

def entity_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    mode: str = "strict",            # "strict" or "partial"
    iou_threshold: float = 0.5,      # for partial: IoU threshold; <=0 means any-overlap
    type_sensitive: bool = True,     # require same entity type to match
) -> Dict[str, Any]:
    """
    Compute entity-level metrics.
    Returns:
      - micro: precision/recall/F1 over all entities
      - macro_f1: average F1 over entity types (per-type macro)
      - per_type: P/R/F1 per entity type (or 'ANY' if type_sensitive=False)
      - counts: tp/fp/fn totals
    """
    assert len(y_true) == len(y_pred), "Mismatched number of sequences"
    tp = fp = fn = 0
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gt_seq, pd_seq in zip(y_true, y_pred):
        gold = [(s, e, t) for (s, e, t) in tags_to_spans(gt_seq)]
        pred = [(s, e, t) for (s, e, t) in tags_to_spans(pd_seq)]
        gold_used = [False] * len(gold)

        for (ps, pe, pt) in pred:
            best_j, best_score = -1, -1.0
            for j, (gs, ge, gt) in enumerate(gold):
                if gold_used[j]:
                    continue
                if type_sensitive and gt != pt:
                    continue

                match_ok, score = False, 0.0
                if mode == "strict":
                    match_ok = (ps == gs and pe == ge)
                    score = 1.0 if match_ok else 0.0
                else:
                    if iou_threshold <= 0:
                        match_ok = _overlap((ps, pe), (gs, ge)) > 0
                        score = _overlap((ps, pe), (gs, ge))
                    else:
                        iou = _iou((ps, pe), (gs, ge))
                        match_ok = iou >= iou_threshold
                        score = iou

                if match_ok and score > best_score:
                    best_score, best_j = score, j

            key = pt if type_sensitive else "ANY"
            if best_j >= 0:
                tp += 1
                gold_used[best_j] = True
                per_type[key]["tp"] += 1
            else:
                fp += 1
                per_type[key]["fp"] += 1

        for j, used in enumerate(gold_used):
            if not used:
                fn += 1
                key = gold[j][2] if type_sensitive else "ANY"
                per_type[key]["fn"] += 1

    micro = _prf(tp, fp, fn)
    type_scores = {t: _prf(d["tp"], d["fp"], d["fn"]) for t, d in per_type.items()}
    macro_f1 = (sum(v["f1"] for v in type_scores.values()) / len(type_scores)) if type_scores else 0.0
    return {"micro": micro, "macro_f1": macro_f1, "per_type": type_scores, "counts": {"tp": tp, "fp": fp, "fn": fn}}

# =========================
# 4) Boundary metrics (Begin/End)
# =========================

def boundary_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    which: str = "begin",            # "begin" or "end"
    type_sensitive: bool = False
) -> Dict[str, float]:
    """Compute boundary-level P/R/F1 for begin or end positions."""
    assert which in {"begin", "end"}
    assert len(y_true) == len(y_pred)
    tp = fp = fn = 0

    for gt_seq, pd_seq in zip(y_true, y_pred):
        gold = tags_to_spans(gt_seq)
        pred = tags_to_spans(pd_seq)
        if which == "begin":
            gold_pts = [(s, t) for (s, e, t) in gold]
            pred_pts = [(s, t) for (s, e, t) in pred]
        else:
            gold_pts = [(e - 1, t) for (s, e, t) in gold]  # end token index
            pred_pts = [(e - 1, t) for (s, e, t) in pred]

        used = [False] * len(gold_pts)
        for p in pred_pts:
            hit = False
            for j, g in enumerate(gold_pts):
                if used[j]:
                    continue
                if (p[0] == g[0]) and ((not type_sensitive) or (p[1] == g[1])):
                    used[j] = True
                    hit = True
                    break
            if hit:
                tp += 1
            else:
                fp += 1
        fn += used.count(False)

    return _prf(tp, fp, fn)

# =========================
# 5) Token-level metrics (reference)
# =========================

def token_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, float]:
    """
    Token accuracy and "entity token" F1 (treat any non-'O' as positive).
    Not recommended as the sole "final" metric for NER, but useful as a sanity check.
    """
    assert len(y_true) == len(y_pred)
    total = correct = 0
    pos_tp = pos_fp = pos_fn = 0

    for gt, pd in zip(y_true, y_pred):
        assert len(gt) == len(pd), "Mismatched sequence lengths"
        for g, p in zip(gt, pd):
            total += 1
            if g == p:
                correct += 1
            g_pos, p_pos = (g != "O"), (p != "O")
            if p_pos and g_pos:
                pos_tp += 1
            elif p_pos and not g_pos:
                pos_fp += 1
            elif (not p_pos) and g_pos:
                pos_fn += 1

    acc = correct / total if total > 0 else 0.0
    prf = _prf(pos_tp, pos_fp, pos_fn)
    return {
        "token_accuracy": acc,
        "entity_token_precision": prf["precision"],
        "entity_token_recall": prf["recall"],
        "entity_token_f1": prf["f1"],
    }

# =========================
# 6) Convenience: compute a set of common metrics at once
# =========================

def compute_all(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    strict_type_sensitive: bool = True,
    partial_iou: float = 0.5,
) -> Dict[str, Any]:
    """Compute strict/partial entity metrics, boundary metrics, and token-level metrics."""
    res = {}
    res["strict"] = entity_metrics(y_true, y_pred, mode="strict", type_sensitive=strict_type_sensitive)
    res["partial_any"] = entity_metrics(y_true, y_pred, mode="partial", iou_threshold=0.0, type_sensitive=False)
    res["partial_iou"] = entity_metrics(y_true, y_pred, mode="partial", iou_threshold=partial_iou, type_sensitive=True)
    res["boundary_begin"] = boundary_metrics(y_true, y_pred, which="begin")
    res["boundary_end"] = boundary_metrics(y_true, y_pred, which="end")
    res["token"] = token_metrics(y_true, y_pred)
    return res

# =========================
# 7) Alignment / conversion helpers
# =========================

def ids_to_tags(
    labels: List[List[int]],
    preds: List[List[int]],
    id2label: Dict[int, str],
    ignore_index: int = -100,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Convert (gold ids, pred ids) into (y_true, y_pred) tag sequences.
    Positions with label == ignore_index (e.g., -100) are skipped (useful for subwords/padding).
    """
    y_true, y_pred = [], []
    for gold_seq, pred_seq in zip(labels, preds):
        t_sent, p_sent = [], []
        for g, p in zip(gold_seq, pred_seq):
            if g == ignore_index:
                continue
            t_sent.append(id2label[int(g)])
            p_sent.append(id2label[int(p)])
        y_true.append(t_sent)
        y_pred.append(p_sent)
    return y_true, y_pred

def logits_to_tag_seqs(
    logits: Any,                 # (N, L, C) ndarray-like (supports .argmax(-1))
    label_ids: Any,              # (N, L)
    id2label: Dict[int, str],
    ignore_index: int = -100,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Convert (logits, label_ids) to (y_true, y_pred) tag sequences.
    Tries to work even if numpy is not installed by falling back to pure Python.
    """
    try:
        import numpy as np  # type: ignore
    except Exception:
        # Fallback: treat logits as nested lists and take argmax per position
        pred_ids = [[max(range(len(vj)), key=lambda c: vj[c]) for vj in vi] for vi in logits]
        labels_list = label_ids
        preds_list = pred_ids
    else:
        pred_ids = np.asarray(logits).argmax(-1)
        labels_list = np.asarray(label_ids).tolist()
        preds_list = np.asarray(pred_ids).tolist()

    return ids_to_tags(labels_list, preds_list, id2label, ignore_index=ignore_index)

def align_with_word_ids(
    pred_ids: Any,                  # (N, L) predicted ids (ndarray or list[list[int]])
    gold_ids: Any,                  # (N, L) gold ids
    word_ids_list: List[List[Optional[int]]],
    id2label: Dict[int, str],
    ignore_index: int = -100,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Align to words using tokenizer.word_ids(): keep only the first subword per word.
    word_ids_list is a list per sample; within each, positions are None for specials,
    or an integer word index for subwords.
    """
    try:
        import numpy as np  # type: ignore
        pred_ids = np.asarray(pred_ids).tolist()
        gold_ids = np.asarray(gold_ids).tolist()
    except Exception:
        pass

    y_true, y_pred = [], []
    for pred_seq, gold_seq, wids in zip(pred_ids, gold_ids, word_ids_list):
        prev_w = None
        t_sent, p_sent = [], []
        for j, w in enumerate(wids):
            if w is None or w == prev_w:
                continue
            prev_w = w
            if gold_seq[j] == ignore_index:
                continue
            t_sent.append(id2label[int(gold_seq[j])])
            p_sent.append(id2label[int(pred_seq[j])])
        y_true.append(t_sent)
        y_pred.append(p_sent)
    return y_true, y_pred

# =========================
# 8) Small self-test (runs if executed directly)
# =========================

if __name__ == "__main__":
    ## example
    y_true_demo = [
        ["B-PER","I-PER","O","S-ORG","O","B-LOC","L-LOC"],
        ["O","B-ORG","E-ORG","O"]
    ]
    y_pred_demo = [
        ["B-PER","I-PER","O","S-ORG","O","B-LOC","L-LOC"],
        ["O","B-ORG","I-ORG","O"]
    ]
    print("Strict metrics:", compute_all(y_true_demo, y_pred_demo)["strict"])

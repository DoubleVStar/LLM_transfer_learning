"""
LLM Classification Script for Biomedical Sentences
-------------------------------------------------

Reads a CSV with columns:
  PMCID | ORIGINAL SENTENCE | NER ANNOTATION | IAO SECTION

Uses an Ollama-served DeepSeek model (e.g. `deepseek-r1:32b`) to classify
each sentence into ONE of 12 predefined study types. Two modes:
  1) text only
  2) text + NER

This version:
  - Uses ONLY the Python client `ollama.chat` (no HTTP / no heuristic fallback).
  - Disables thinking output (think=False).
  - Enforces JSON structured output via JSON Schema (format=SCHEMA).
  - Returns exactly one label from the 12 categories.

Output columns:
  PMCID | ORIGINAL SENTENCE | NER ANNOTATION | LLM CLASSIFICATION RESULT | IAO SECTION

Usage:
  python llm_classifier_1.py --input '/home/kevin/MultiTagger-v2/DATA/fulltext_json/distribution_dir/IAO_selected_ner.csv' --output '/home/kevin/MultiTagger-v2/DATA/fulltext_json/distribution_dir/LLM_text_cls.csv' --use_ner

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional

import pandas as pd
from ollama import chat, ChatResponse  # pip install ollama

# -----------------------------------------------------------------------------
# Categories
# -----------------------------------------------------------------------------
CATEGORIES: List[str] = [
    "Meta-analysis",
    "Systematic analysis",
    "Randomised controlled trial",
    "Non-randomised controlled trial",
    "Cohort study",
    "Case-control study",
    "Cross-sectional study",
    "Case series",
    "Case report",
    "Narrative review, Expert opinion, Editorial, Ideas",
    "Animal research, In vivo studies",
    "In vitro research, Laboratory research",
]

# -----------------------------------------------------------------------------
# System Prompt (system prompt generator 风格)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT: str = (
    "你是一位医学研究方法论专家与文本分类助手。"
    "任务：对给定句子（可含NER标注）进行研究类型分类，并且从以下列表中选择且仅选择一个最合适的类别："
    + "，".join(CATEGORIES)
    + "。"
    "原则："
    "1) 仅返回最终结论，不要解释、不要编号、不要思考过程；"
    "2) 如句子涉及多要素，选择最具代表性的单一类别；"
    "3) 如信息有限，基于可得线索做出最合理判断；"
    "4) 输出必须是严格的 JSON，对象形如：{\"classification\": \"<类别>\"}，其中 <类别> 必须严格等于上述列表之一。"
)

# 结构化输出 Schema（强制仅返回一个字段 `classification` 且取值必须在 12 类中）
SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": CATEGORIES}
    },
    "required": ["classification"],
    "additionalProperties": False,
}

# 若模型仍返回了 <think>，做一次兜底清洗（用于历史/异常输出）
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def build_prompt(text: str, ner: str, use_ner: bool) -> str:
    """构造给模型的用户消息（不再重复任务规则与分类列表，集中在具体内容）。"""
    if use_ner:
        return f"句子：{text}\nNER标注：{ner}"
    return f"句子：{text}"


def _normalize_to_category(candidate: str) -> str:
    """将模型返回的文本归一到 12 个合法标签之一（大小写不敏感精确匹配）。"""
    cand = (candidate or "").strip()
    if not cand:
        return ""
    lc = cand.casefold()
    for c in CATEGORIES:
        if lc == c.casefold():
            return c
    # 非法字符串，返回原文（上游已尽量用 JSON 格式避免）
    return cand


def _clean_think_and_extract(raw: str) -> str:
    """去除 <think>...</think> 并尝试用最后一行作为候选标签。"""
    s = (raw or "").strip()
    if not s:
        return ""
    s = _THINK_RE.sub("", s).strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    return _normalize_to_category(lines[-1])


def call_ollama_classify(prompt: str, model: str) -> str:
    """
    仅使用 Python 客户端 `ollama.chat`：
      - think=False 禁用思考输出
      - format=SCHEMA 强制 JSON 结构
      - options.temperature=0 提高确定性
    返回严格的 12 类标签之一；若解析失败则尝试 <think> 清洗兜底。
    """
    resp: ChatResponse = chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        think=False,         # 关闭 <think>
        stream=False,
        format=SCHEMA,       # 强制返回 {"classification": "..."}
        options={"temperature": 0}
    )

    content = getattr(resp.message, "content", None)
    if isinstance(content, str):
        # 期望为 JSON，解析后取 classification
        try:
            obj = json.loads(content)
            label = obj.get("classification", "")
            return _normalize_to_category(label)
        except Exception:
            # 如果不是 JSON（少数模型/版本可能无视 format），尝试清洗 <think>
            return _clean_think_and_extract(content)
    return ""


def classify_row(text: str, ner: str, use_ner: bool, model: str) -> str:
    """对单条句子进行分类（仅用 `ollama.chat`，无HTTP/无启发式回退）。"""
    prompt = build_prompt(text, ner, use_ner)
    return call_ollama_classify(prompt, model)


def process_dataframe(df: pd.DataFrame, use_ner: bool, model: str) -> pd.DataFrame:
    """逐行分类并新增列 `LLM CLASSIFICATION RESULT`。"""
    required_cols = {"PMCID", "ORIGINAL SENTENCE", "NER ANNOTATION", "IAO SECTION"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    df = df.copy()
    results: List[str] = []
    total = len(df)

    for i, row in df.iterrows():
        text = str(row["ORIGINAL SENTENCE"])
        ner = str(row["NER ANNOTATION"])
        label = classify_row(text, ner, use_ner, model)
        results.append(label)
        # print(f"[process_dataframe] Row {len(results)}/{total} classified as: {label}")
        print(f"Row {len(results)}/{total}")

    df["LLM CLASSIFICATION RESULT"] = results
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify biomedical sentences into study types via DeepSeek (Ollama)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with columns: PMCID, ORIGINAL SENTENCE, NER ANNOTATION, IAO SECTION",
    )
    parser.add_argument(
        "--output",
        default="classified_output.csv",
        help="Path to save CSV with classification results",
    )
    parser.add_argument(
        "--use_ner",
        action="store_true",
        help="Include NER annotations in the prompt",
    )
    parser.add_argument(
        "--model",
        default="deepseek-r1:32b",
        help="Ollama model name (e.g. deepseek-r1:32b)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit rows for quick testing",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8")
    if args.limit is not None:
        df = df.iloc[: args.limit].copy()

    out_df = process_dataframe(df, args.use_ner, args.model)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[main] Classification complete. Results saved to: {args.output}")
    print(out_df.head())


if __name__ == "__main__":
    main()

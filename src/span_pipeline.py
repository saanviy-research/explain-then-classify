"""Idea 5: explanation-first pipeline. Stage 1 extracts the minimal evidence
span for lonesomeness (or NONE); stage 2 judges Present/Absent from ONLY the
span + one context sentence, not the full post.
"""
import re

import pandas as pd

import config

_EXTRACT_TEMPLATE = (config.PROMPTS_DIR / "span_extract.txt").read_text(encoding="utf-8")
_JUDGE_TEMPLATE = (config.PROMPTS_DIR / "span_judge.txt").read_text(encoding="utf-8")

_CALIBRATION_SEED = 42
_N_CALIBRATION = 5


def get_calibration_examples(train_df: pd.DataFrame, n: int = _N_CALIBRATION) -> pd.DataFrame:
    """Fixed set of Present train posts with gold spans, for few-shot calibration."""
    present = train_df[train_df["label"] == 1].dropna(subset=["explanation"])
    return present.sample(n, random_state=_CALIBRATION_SEED)


def build_extract_examples_block(train_df: pd.DataFrame) -> str:
    examples = get_calibration_examples(train_df)
    lines = []
    for i, (_, r) in enumerate(examples.iterrows(), 1):
        lines.append(f'{i}. Gold span: "{r["explanation"]}"')
    return "\n".join(lines)


def build_extract_prompt(post_text: str, examples_block: str) -> str:
    return _EXTRACT_TEMPLATE.format(post_text=post_text, examples_block=examples_block)


def build_judge_examples_block(train_df: pd.DataFrame) -> str:
    """A few Present (gold span) and hard Absent (cue-word-adjacent) examples for judge calibration."""
    present = train_df[train_df["label"] == 1].dropna(subset=["explanation"]).sample(
        3, random_state=_CALIBRATION_SEED
    )
    lines = []
    for _, r in present.iterrows():
        lines.append(f'Span: "{r["explanation"]}"  ->  Present')
    # Hard-negative calibration: fixed hand-picked cases (not from train, illustrative only)
    lines.append('Span: "no one would be pay for her assisted living"  ->  Absent (about a dependent, not the author\'s own isolation)')
    lines.append('Span: "everyone else was out with their friends"  ->  Absent (situational, no void/loss framing)')
    return "\n".join(lines)


def build_judge_prompt(span: str, context: str, examples_block: str) -> str:
    return _JUDGE_TEMPLATE.format(span=span, context=context, examples_block=examples_block)


def extract_context_sentence(post_text: str, span: str) -> str:
    """The sentence containing the span, or the span itself if not found verbatim."""
    if not span or span.strip().upper() == "NONE":
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", post_text)
    for s in sentences:
        if span.lower() in s.lower():
            return s.strip()
    return span


def token_recall(extracted: str, gold: str) -> float:
    """Fraction of gold-span tokens present in the extracted span (case-insensitive)."""
    gold_tokens = set(re.findall(r"\w+", gold.lower()))
    if not gold_tokens:
        return 0.0
    extracted_tokens = set(re.findall(r"\w+", extracted.lower()))
    return len(gold_tokens & extracted_tokens) / len(gold_tokens)


def token_precision(extracted: str, gold: str) -> float:
    """Fraction of extracted-span tokens present in the gold span (case-insensitive)."""
    extracted_tokens = set(re.findall(r"\w+", extracted.lower()))
    if not extracted_tokens:
        return 0.0
    gold_tokens = set(re.findall(r"\w+", gold.lower()))
    return len(gold_tokens & extracted_tokens) / len(extracted_tokens)


def token_f1(extracted: str, gold: str) -> float:
    p = token_precision(extracted, gold)
    r = token_recall(extracted, gold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

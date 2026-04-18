"""Scoring implementations: exact, fuzzy, llm_judge."""

from __future__ import annotations

from rapidfuzz import fuzz

from .ollama import generate
from .schema import EvalCase, ScoringMethod

PASS_THRESHOLD = 0.8


def score_case(case: EvalCase, actual: str, ollama_base_url: str = "http://localhost:11434") -> float:
    if case.scoring == ScoringMethod.exact:
        return _exact(case.expected, actual)
    if case.scoring == ScoringMethod.fuzzy:
        return _fuzzy(case.expected, actual)
    if case.scoring == ScoringMethod.llm_judge:
        return _llm_judge(case, actual, ollama_base_url)
    raise ValueError(f"Unknown scoring method: {case.scoring}")


def is_pass(score: float) -> bool:
    return score >= PASS_THRESHOLD


def _exact(expected: str, actual: str) -> float:
    return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0


def _fuzzy(expected: str, actual: str) -> float:
    return fuzz.token_set_ratio(expected, actual) / 100.0


def _llm_judge(case: EvalCase, actual: str, base_url: str) -> float:
    judge_model = case.judge_model or "llama3"
    prompt = (
        "You are an impartial judge evaluating an AI answer.\n\n"
        f"Question: {case.prompt}\n\n"
        f"Expected answer: {case.expected}\n\n"
        f"Actual answer: {actual}\n\n"
        "Rate the actual answer on a scale from 0.0 to 1.0 where 1.0 means perfectly correct "
        "and 0.0 means completely wrong. Reply with ONLY a decimal number, nothing else."
    )
    try:
        response = generate(judge_model, prompt, base_url=base_url)
        return max(0.0, min(1.0, float(response.strip())))
    except (ValueError, Exception):
        return 0.0

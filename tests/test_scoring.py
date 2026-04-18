"""Tests for scoring functions."""

from unittest.mock import patch

import pytest

from model_eval.schema import EvalCase, ScoringMethod
from model_eval.scoring import PASS_THRESHOLD, is_pass, score_case


def make_case(scoring: ScoringMethod, expected: str = "hello world", judge_model: str | None = None):
    return EvalCase(prompt="test", expected=expected, scoring=scoring, judge_model=judge_model)


# --- exact ---

def test_exact_pass():
    case = make_case(ScoringMethod.exact, "hello world")
    assert score_case(case, "hello world") == 1.0


def test_exact_case_insensitive():
    case = make_case(ScoringMethod.exact, "Hello World")
    assert score_case(case, "hello world") == 1.0


def test_exact_strips_whitespace():
    case = make_case(ScoringMethod.exact, "hello")
    assert score_case(case, "  hello  ") == 1.0


def test_exact_fail():
    case = make_case(ScoringMethod.exact, "hello")
    assert score_case(case, "world") == 0.0


# --- fuzzy ---

def test_fuzzy_identical():
    case = make_case(ScoringMethod.fuzzy, "Paris is the capital of France")
    score = score_case(case, "Paris is the capital of France")
    assert score == 1.0


def test_fuzzy_similar():
    case = make_case(ScoringMethod.fuzzy, "capital of France")
    score = score_case(case, "Paris is the capital of France, a beautiful city")
    assert score >= 0.5


def test_fuzzy_unrelated():
    case = make_case(ScoringMethod.fuzzy, "capital of France")
    score = score_case(case, "banana smoothie recipe")
    assert score < 0.5


# --- llm_judge ---

def test_llm_judge_success():
    case = make_case(ScoringMethod.llm_judge, judge_model="llama3")
    with patch("model_eval.scoring.generate", return_value="0.9") as mock_gen:
        score = score_case(case, "some answer")
    assert abs(score - 0.9) < 1e-9
    mock_gen.assert_called_once()


def test_llm_judge_clamps_to_range():
    case = make_case(ScoringMethod.llm_judge, judge_model="llama3")
    with patch("model_eval.scoring.generate", return_value="1.5"):
        assert score_case(case, "x") == 1.0
    with patch("model_eval.scoring.generate", return_value="-0.5"):
        assert score_case(case, "x") == 0.0


def test_llm_judge_non_numeric_returns_zero():
    case = make_case(ScoringMethod.llm_judge, judge_model="llama3")
    with patch("model_eval.scoring.generate", return_value="not a number"):
        assert score_case(case, "x") == 0.0


def test_llm_judge_generate_raises_returns_zero():
    case = make_case(ScoringMethod.llm_judge, judge_model="llama3")
    with patch("model_eval.scoring.generate", side_effect=Exception("connection refused")):
        assert score_case(case, "x") == 0.0


def test_llm_judge_uses_default_model_when_none():
    case = EvalCase(prompt="q", expected="e", scoring=ScoringMethod.llm_judge, judge_model=None)
    with patch("model_eval.scoring.generate", return_value="0.7") as mock_gen:
        score_case(case, "answer")
    call_args = mock_gen.call_args
    assert call_args[0][0] == "llama3"  # default model


# --- is_pass ---

def test_is_pass_above_threshold():
    assert is_pass(PASS_THRESHOLD) is True
    assert is_pass(1.0) is True


def test_is_pass_below_threshold():
    assert is_pass(PASS_THRESHOLD - 0.01) is False
    assert is_pass(0.0) is False


# --- unknown scoring method ---

def test_unknown_scoring_raises():
    case = EvalCase(prompt="q", expected="e", scoring="badmethod")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        score_case(case, "answer")

"""Tests for suite runner."""

from unittest.mock import patch

from model_eval.runner import run_suite
from model_eval.schema import EvalCase, EvalSuite, ScoringMethod


def make_suite(*cases: EvalCase) -> EvalSuite:
    return EvalSuite(name="test", cases=list(cases))


def test_run_suite_all_pass():
    suite = make_suite(EvalCase(prompt="q1", expected="a1"), EvalCase(prompt="q2", expected="a2"))
    with patch("model_eval.runner.generate", side_effect=["a1", "a2"]):
        result = run_suite(suite, "llama3")
    assert result.passed == 2
    assert result.total == 2


def test_run_suite_partial_pass():
    suite = make_suite(EvalCase(prompt="q1", expected="yes"), EvalCase(prompt="q2", expected="yes"))
    with patch("model_eval.runner.generate", side_effect=["yes", "no"]):
        result = run_suite(suite, "llama3")
    assert result.passed == 1


def test_run_suite_generate_error_counted_as_fail():
    suite = make_suite(EvalCase(prompt="q", expected="a"))
    with patch("model_eval.runner.generate", side_effect=Exception("timeout")):
        result = run_suite(suite, "llama3")
    assert result.passed == 0
    assert result.case_results[0].error == "timeout"
    assert result.case_results[0].score == 0.0


def test_run_suite_on_progress_callback():
    suite = make_suite(EvalCase(prompt="q", expected="a"))
    received = []
    with patch("model_eval.runner.generate", return_value="a"):
        run_suite(suite, "llama3", on_progress=received.append)
    assert len(received) == 1
    assert received[0].passed is True


def test_run_suite_model_stored():
    suite = make_suite(EvalCase(prompt="q", expected="a"))
    with patch("model_eval.runner.generate", return_value="a"):
        result = run_suite(suite, "mistral")
    assert result.model == "mistral"


def test_run_suite_fuzzy_scoring():
    suite = make_suite(
        EvalCase(prompt="q", expected="capital of France", scoring=ScoringMethod.fuzzy)
    )
    with patch("model_eval.runner.generate", return_value="Paris is the capital of France"):
        result = run_suite(suite, "llama3")
    assert result.case_results[0].score > 0.5

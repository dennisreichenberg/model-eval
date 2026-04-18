"""Tests for schema dataclasses."""

from model_eval.schema import CaseResult, EvalCase, EvalSuite, ScoringMethod, SuiteResult


def _make_suite() -> EvalSuite:
    return EvalSuite(
        name="test-suite",
        cases=[
            EvalCase(prompt="p1", expected="e1"),
            EvalCase(prompt="p2", expected="e2"),
        ],
    )


def test_suite_result_pass_count():
    suite = _make_suite()
    result = SuiteResult(
        suite=suite,
        model="llama3",
        case_results=[
            CaseResult(case=suite.cases[0], actual="e1", score=1.0, passed=True),
            CaseResult(case=suite.cases[1], actual="wrong", score=0.0, passed=False),
        ],
    )
    assert result.passed == 1
    assert result.total == 2
    assert result.pass_rate == 0.5


def test_suite_result_avg_score():
    suite = _make_suite()
    result = SuiteResult(
        suite=suite,
        model="llama3",
        case_results=[
            CaseResult(case=suite.cases[0], actual="e1", score=0.8, passed=True),
            CaseResult(case=suite.cases[1], actual="e2", score=0.6, passed=False),
        ],
    )
    assert abs(result.avg_score - 0.7) < 1e-9


def test_suite_result_empty():
    suite = EvalSuite(name="empty", cases=[])
    result = SuiteResult(suite=suite, model="llama3")
    assert result.pass_rate == 0.0
    assert result.avg_score == 0.0
    assert result.total == 0


def test_eval_case_defaults():
    case = EvalCase(prompt="q", expected="a")
    assert case.scoring == ScoringMethod.exact
    assert case.judge_model is None
    assert case.description is None


def test_scoring_method_values():
    assert ScoringMethod("exact") == ScoringMethod.exact
    assert ScoringMethod("fuzzy") == ScoringMethod.fuzzy
    assert ScoringMethod("llm_judge") == ScoringMethod.llm_judge

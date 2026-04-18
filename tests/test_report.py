"""Tests for report rendering and JSON export."""

import json
from pathlib import Path

from model_eval.report import print_compare_results, print_suite_result, write_json_report
from model_eval.schema import CaseResult, EvalCase, EvalSuite, SuiteResult


def make_result(model: str = "llama3", passed: bool = True) -> SuiteResult:
    suite = EvalSuite(name="test", cases=[EvalCase(prompt="q", expected="a")])
    score = 1.0 if passed else 0.0
    return SuiteResult(
        suite=suite,
        model=model,
        case_results=[CaseResult(case=suite.cases[0], actual="a" if passed else "x", score=score, passed=passed)],
    )


def test_write_json_report_structure(tmp_path: Path):
    result = make_result("llama3", passed=True)
    out = tmp_path / "report.json"
    write_json_report([result], out)
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["model"] == "llama3"
    assert data[0]["passed"] == 1
    assert data[0]["total"] == 1
    assert "cases" in data[0]
    assert len(data[0]["cases"]) == 1


def test_write_json_report_failed_case(tmp_path: Path):
    result = make_result("mistral", passed=False)
    out = tmp_path / "report.json"
    write_json_report([result], out)
    data = json.loads(out.read_text())
    assert data[0]["passed"] == 0
    assert data[0]["cases"][0]["passed"] is False


def test_write_json_report_multiple_models(tmp_path: Path):
    results = [make_result("llama3"), make_result("mistral", passed=False)]
    out = tmp_path / "report.json"
    write_json_report(results, out)
    data = json.loads(out.read_text())
    assert len(data) == 2
    models = {r["model"] for r in data}
    assert models == {"llama3", "mistral"}


def test_write_json_report_with_error(tmp_path: Path):
    suite = EvalSuite(name="err-suite", cases=[EvalCase(prompt="q", expected="a")])
    case_result = CaseResult(case=suite.cases[0], actual="", score=0.0, passed=False, error="timeout")
    result = SuiteResult(suite=suite, model="llama3", case_results=[case_result])
    out = tmp_path / "report.json"
    write_json_report([result], out)
    data = json.loads(out.read_text())
    assert data[0]["cases"][0]["error"] == "timeout"


def test_print_suite_result_no_exception():
    result = make_result()
    print_suite_result(result)  # must not raise


def test_print_compare_results_no_exception():
    results = [make_result("llama3"), make_result("mistral", passed=False)]
    print_compare_results(results)  # must not raise

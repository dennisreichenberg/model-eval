"""Tests for CLI commands via click test runner."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from model_eval.cli import main
from model_eval.schema import CaseResult, EvalCase, EvalSuite, SuiteResult


def make_suite_result(model: str = "llama3") -> SuiteResult:
    suite = EvalSuite(name="demo", cases=[EvalCase(prompt="q", expected="a")])
    return SuiteResult(
        suite=suite,
        model=model,
        case_results=[CaseResult(case=suite.cases[0], actual="a", score=1.0, passed=True)],
    )


def write_eval_yaml(tmp_path: Path, name: str = "demo") -> Path:
    p = tmp_path / "eval.yaml"
    p.write_text(
        textwrap.dedent(f"""
            name: {name}
            cases:
              - prompt: "q"
                expected: "a"
        """),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def runner():
    return CliRunner()


# --- run command ---

def test_run_cmd_success(runner, tmp_path):
    yaml_file = write_eval_yaml(tmp_path)
    with patch("model_eval.cli.run_suite", return_value=make_suite_result()) as mock_run:
        result = runner.invoke(main, ["run", str(yaml_file), "--model", "llama3"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_run_cmd_missing_eval_file(runner, tmp_path):
    result = runner.invoke(main, ["run", str(tmp_path / "nonexistent.yaml"), "--model", "llama3"])
    assert result.exit_code != 0


def test_run_cmd_writes_json_report(runner, tmp_path):
    yaml_file = write_eval_yaml(tmp_path)
    report_file = tmp_path / "report.json"
    with patch("model_eval.cli.run_suite", return_value=make_suite_result()):
        result = runner.invoke(
            main, ["run", str(yaml_file), "--model", "llama3", "--json-report", str(report_file)]
        )
    assert result.exit_code == 0
    assert report_file.exists()
    data = json.loads(report_file.read_text())
    assert len(data) == 1


def test_run_cmd_missing_model_flag(runner, tmp_path):
    yaml_file = write_eval_yaml(tmp_path)
    result = runner.invoke(main, ["run", str(yaml_file)])
    assert result.exit_code != 0


# --- compare command ---

def test_compare_cmd_success(runner, tmp_path):
    yaml_file = write_eval_yaml(tmp_path)
    with patch(
        "model_eval.cli.run_suite",
        side_effect=[make_suite_result("llama3"), make_suite_result("mistral")],
    ):
        result = runner.invoke(
            main, ["compare", "--models", "llama3,mistral", "--eval", str(yaml_file)]
        )
    assert result.exit_code == 0


def test_compare_cmd_writes_json_report(runner, tmp_path):
    yaml_file = write_eval_yaml(tmp_path)
    report_file = tmp_path / "compare.json"
    with patch(
        "model_eval.cli.run_suite",
        side_effect=[make_suite_result("llama3"), make_suite_result("mistral")],
    ):
        runner.invoke(
            main,
            ["compare", "--models", "llama3,mistral", "--eval", str(yaml_file), "--json-report", str(report_file)],
        )
    assert report_file.exists()
    data = json.loads(report_file.read_text())
    assert len(data) == 2


def test_compare_cmd_missing_eval(runner, tmp_path):
    result = runner.invoke(
        main, ["compare", "--models", "llama3", "--eval", str(tmp_path / "missing.yaml")]
    )
    assert result.exit_code != 0


def test_compare_cmd_empty_models(runner, tmp_path):
    yaml_file = write_eval_yaml(tmp_path)
    result = runner.invoke(main, ["compare", "--models", "", "--eval", str(yaml_file)])
    assert result.exit_code != 0

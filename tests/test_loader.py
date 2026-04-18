"""Tests for YAML eval suite loader."""

import textwrap
from pathlib import Path

import pytest

from model_eval.loader import load_suite
from model_eval.schema import ScoringMethod


def write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "suite.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_load_basic_suite(tmp_path):
    p = write_yaml(tmp_path, """
        name: basic
        cases:
          - prompt: "What is 2+2?"
            expected: "4"
    """)
    suite = load_suite(p)
    assert suite.name == "basic"
    assert len(suite.cases) == 1
    assert suite.cases[0].prompt == "What is 2+2?"
    assert suite.cases[0].expected == "4"
    assert suite.cases[0].scoring == ScoringMethod.exact


def test_load_fuzzy_scoring(tmp_path):
    p = write_yaml(tmp_path, """
        name: fuzzy-suite
        cases:
          - prompt: "Describe Paris"
            expected: "capital of France"
            scoring: fuzzy
    """)
    suite = load_suite(p)
    assert suite.cases[0].scoring == ScoringMethod.fuzzy


def test_load_llm_judge_scoring(tmp_path):
    p = write_yaml(tmp_path, """
        name: judge-suite
        default_judge_model: mistral
        cases:
          - prompt: "What is Python?"
            expected: "a programming language"
            scoring: llm_judge
    """)
    suite = load_suite(p)
    assert suite.cases[0].scoring == ScoringMethod.llm_judge
    assert suite.cases[0].judge_model == "mistral"
    assert suite.default_judge_model == "mistral"


def test_load_per_case_judge_model(tmp_path):
    p = write_yaml(tmp_path, """
        name: per-case-judge
        default_judge_model: llama3
        cases:
          - prompt: "q"
            expected: "a"
            scoring: llm_judge
            judge_model: phi3
    """)
    suite = load_suite(p)
    assert suite.cases[0].judge_model == "phi3"


def test_load_description_field(tmp_path):
    p = write_yaml(tmp_path, """
        name: desc-suite
        cases:
          - prompt: "q"
            expected: "a"
            description: "sanity check"
    """)
    suite = load_suite(p)
    assert suite.cases[0].description == "sanity check"


def test_load_name_fallback_to_stem(tmp_path):
    p = tmp_path / "my_suite.yaml"
    p.write_text("cases:\n  - prompt: q\n    expected: a\n", encoding="utf-8")
    suite = load_suite(p)
    assert suite.name == "my_suite"


def test_load_unknown_scoring_raises(tmp_path):
    p = write_yaml(tmp_path, """
        name: bad
        cases:
          - prompt: q
            expected: a
            scoring: unknown_method
    """)
    with pytest.raises(ValueError, match="Unknown scoring method"):
        load_suite(p)


def test_load_multiple_cases(tmp_path):
    p = write_yaml(tmp_path, """
        name: multi
        cases:
          - prompt: "q1"
            expected: "a1"
          - prompt: "q2"
            expected: "a2"
            scoring: fuzzy
          - prompt: "q3"
            expected: "a3"
            scoring: exact
    """)
    suite = load_suite(p)
    assert len(suite.cases) == 3
    assert suite.cases[1].scoring == ScoringMethod.fuzzy

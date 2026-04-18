"""Data models for eval suites and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ScoringMethod(str, Enum):
    exact = "exact"
    fuzzy = "fuzzy"
    llm_judge = "llm_judge"


@dataclass
class EvalCase:
    prompt: str
    expected: str
    scoring: ScoringMethod = ScoringMethod.exact
    judge_model: Optional[str] = None
    description: Optional[str] = None


@dataclass
class EvalSuite:
    name: str
    cases: list[EvalCase] = field(default_factory=list)
    default_judge_model: Optional[str] = None


@dataclass
class CaseResult:
    case: EvalCase
    actual: str
    score: float  # 0.0 - 1.0
    passed: bool
    error: Optional[str] = None


@dataclass
class SuiteResult:
    suite: EvalSuite
    model: str
    case_results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.case_results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_score(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.score for r in self.case_results) / len(self.case_results)

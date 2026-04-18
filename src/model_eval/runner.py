"""Run an eval suite against one model and collect results."""

from __future__ import annotations

from .ollama import generate
from .schema import CaseResult, EvalSuite, SuiteResult
from .scoring import is_pass, score_case


def run_suite(
    suite: EvalSuite,
    model: str,
    ollama_base_url: str = "http://localhost:11434",
    on_progress=None,
) -> SuiteResult:
    result = SuiteResult(suite=suite, model=model)

    for case in suite.cases:
        error = None
        actual = ""
        try:
            actual = generate(model, case.prompt, base_url=ollama_base_url)
            score = score_case(case, actual, ollama_base_url)
        except Exception as exc:
            score = 0.0
            error = str(exc)

        case_result = CaseResult(
            case=case,
            actual=actual,
            score=score,
            passed=is_pass(score),
            error=error,
        )
        result.case_results.append(case_result)

        if on_progress:
            on_progress(case_result)

    return result

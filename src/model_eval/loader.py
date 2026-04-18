"""YAML eval suite loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import EvalCase, EvalSuite, ScoringMethod


def load_suite(path: str | Path) -> EvalSuite:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)

    name = data.get("name", path.stem)
    default_judge = data.get("default_judge_model")
    cases: list[EvalCase] = []

    for item in data.get("cases", []):
        scoring_raw = item.get("scoring", "exact")
        try:
            scoring = ScoringMethod(scoring_raw)
        except ValueError:
            raise ValueError(f"Unknown scoring method '{scoring_raw}' in {path}")

        judge = item.get("judge_model") or default_judge
        cases.append(
            EvalCase(
                prompt=item["prompt"],
                expected=item["expected"],
                scoring=scoring,
                judge_model=judge,
                description=item.get("description"),
            )
        )

    return EvalSuite(name=name, cases=cases, default_judge_model=default_judge)

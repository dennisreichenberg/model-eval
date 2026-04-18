"""Render results as Rich table and optional JSON report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.table import Table

from .schema import SuiteResult

console = Console()


def print_suite_result(result: SuiteResult) -> None:
    table = Table(title=f"Eval: {result.suite.name}  |  Model: {result.model}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Prompt", overflow="fold", max_width=40)
    table.add_column("Expected", overflow="fold", max_width=30)
    table.add_column("Actual", overflow="fold", max_width=30)
    table.add_column("Score", justify="right")
    table.add_column("Pass", justify="center")

    for i, cr in enumerate(result.case_results, 1):
        pass_str = "[green]yes[/green]" if cr.passed else "[red]no[/red]"
        actual_display = cr.error or cr.actual
        table.add_row(
            str(i),
            cr.case.prompt[:80],
            cr.case.expected[:50],
            actual_display[:50],
            f"{cr.score:.2f}",
            pass_str,
        )

    console.print(table)
    console.print(
        f"  Passed: [bold]{result.passed}/{result.total}[/bold]  "
        f"Pass rate: [bold]{result.pass_rate:.0%}[/bold]  "
        f"Avg score: [bold]{result.avg_score:.2f}[/bold]"
    )


def print_compare_results(results: Sequence[SuiteResult]) -> None:
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Pass", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Pass rate", justify="right")
    table.add_column("Avg score", justify="right")

    for r in results:
        table.add_row(
            r.model,
            str(r.passed),
            str(r.total),
            f"{r.pass_rate:.0%}",
            f"{r.avg_score:.2f}",
        )

    console.print(table)


def write_json_report(results: Sequence[SuiteResult], path: str | Path) -> None:
    data = []
    for r in results:
        data.append(
            {
                "model": r.model,
                "suite": r.suite.name,
                "passed": r.passed,
                "total": r.total,
                "pass_rate": round(r.pass_rate, 4),
                "avg_score": round(r.avg_score, 4),
                "cases": [
                    {
                        "prompt": cr.case.prompt,
                        "expected": cr.case.expected,
                        "actual": cr.actual,
                        "scoring": cr.case.scoring.value,
                        "score": round(cr.score, 4),
                        "passed": cr.passed,
                        "error": cr.error,
                    }
                    for cr in r.case_results
                ],
            }
        )
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

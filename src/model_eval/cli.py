"""CLI entry point for model-eval."""

from __future__ import annotations

import sys

import click
from rich.console import Console

from .loader import load_suite
from .report import print_compare_results, print_suite_result, write_json_report
from .runner import run_suite

console = Console()
err_console = Console(stderr=True)


@click.group()
@click.version_option()
def main() -> None:
    """Evaluate local LLM answer quality via Ollama."""


@main.command("run")
@click.argument("eval_file", metavar="EVAL_FILE")
@click.option("--model", "-m", required=True, help="Ollama model name, e.g. llama3")
@click.option("--ollama-url", default="http://localhost:11434", show_default=True)
@click.option("--json-report", default=None, help="Optional path to write JSON report")
def run_cmd(eval_file: str, model: str, ollama_url: str, json_report: str | None) -> None:
    """Run an eval suite against a single model."""
    try:
        suite = load_suite(eval_file)
    except Exception as exc:
        err_console.print(f"[red]Failed to load eval file:[/red] {exc}")
        sys.exit(1)

    console.print(f"Running [bold]{suite.name}[/bold] against [cyan]{model}[/cyan] ...")
    result = run_suite(suite, model, ollama_base_url=ollama_url)
    print_suite_result(result)

    if json_report:
        write_json_report([result], json_report)
        console.print(f"JSON report written to [green]{json_report}[/green]")


@main.command("compare")
@click.option("--models", "-M", required=True, help="Comma-separated model names, e.g. llama3,mistral")
@click.option("--eval", "eval_file", required=True, help="Path to eval YAML file")
@click.option("--ollama-url", default="http://localhost:11434", show_default=True)
@click.option("--json-report", default=None, help="Optional path to write JSON report")
def compare_cmd(models: str, eval_file: str, ollama_url: str, json_report: str | None) -> None:
    """Compare multiple models on the same eval suite."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        err_console.print("[red]--models requires at least one model name[/red]")
        sys.exit(1)

    try:
        suite = load_suite(eval_file)
    except Exception as exc:
        err_console.print(f"[red]Failed to load eval file:[/red] {exc}")
        sys.exit(1)

    results = []
    for model in model_list:
        console.print(f"Running [bold]{suite.name}[/bold] against [cyan]{model}[/cyan] ...")
        result = run_suite(suite, model, ollama_base_url=ollama_url)
        print_suite_result(result)
        results.append(result)

    console.print()
    print_compare_results(results)

    if json_report:
        write_json_report(results, json_report)
        console.print(f"JSON report written to [green]{json_report}[/green]")

# model-eval

Evaluate local LLM answer quality via [Ollama](https://ollama.ai). Complements `llm-bench` (which measures throughput/latency) by focusing on *answer correctness*.

## Scoring methods

| Method      | Description                                              |
|-------------|----------------------------------------------------------|
| `exact`     | Case-insensitive string equality                         |
| `fuzzy`     | Token-set ratio via rapidfuzz (threshold >= 0.8 = pass) |
| `llm_judge` | Asks another LLM to rate the answer 0.0-1.0              |

## Eval file format

```yaml
name: coding-basics
default_judge_model: llama3  # optional, used when scoring: llm_judge and no per-case judge_model

cases:
  - prompt: "What keyword defines a function in Python?"
    expected: "def"
    scoring: exact

  - prompt: "Describe what ls does"
    expected: "lists files and directories"
    scoring: fuzzy

  - prompt: "Explain list comprehensions"
    expected: "a concise way to create lists with optional filtering"
    scoring: llm_judge
    judge_model: mistral   # optional per-case override
    description: "sanity check"  # optional label
```

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Run a single eval suite

```bash
model-eval run evals/coding.yaml --model llama3
```

With JSON report:

```bash
model-eval run evals/coding.yaml --model llama3 --json-report results.json
```

### Compare multiple models

```bash
model-eval compare --models llama3,mistral --eval evals/logic.yaml
```

With JSON report:

```bash
model-eval compare --models llama3,mistral --eval evals/logic.yaml --json-report compare.json
```

### Custom Ollama URL

```bash
model-eval run evals/coding.yaml --model llama3 --ollama-url http://my-server:11434
```

## Running tests

```bash
pytest --cov=model_eval --cov-report=term-missing
```

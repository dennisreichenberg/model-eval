"""Ollama API client (generate endpoint)."""

from __future__ import annotations

import httpx


DEFAULT_BASE_URL = "http://localhost:11434"


def generate(
    model: str,
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 120.0,
) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = httpx.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"]

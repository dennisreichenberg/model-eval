"""Tests for Ollama client."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from model_eval.ollama import generate


def test_generate_returns_response():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "42"}
    mock_resp.raise_for_status = MagicMock()

    with patch("model_eval.ollama.httpx.post", return_value=mock_resp) as mock_post:
        result = generate("llama3", "What is 6x7?")

    assert result == "42"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["model"] == "llama3"
    assert call_kwargs[1]["json"]["stream"] is False


def test_generate_raises_on_http_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=MagicMock()
    )

    with patch("model_eval.ollama.httpx.post", return_value=mock_resp):
        with pytest.raises(httpx.HTTPStatusError):
            generate("llama3", "q")


def test_generate_uses_custom_base_url():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "ok"}
    mock_resp.raise_for_status = MagicMock()

    with patch("model_eval.ollama.httpx.post", return_value=mock_resp) as mock_post:
        generate("llama3", "q", base_url="http://custom:12345")

    url = mock_post.call_args[0][0]
    assert url.startswith("http://custom:12345")

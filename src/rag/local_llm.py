"""
Local LLM client (Ollama, OpenAI-compatible endpoint).

All generation in this project goes through this one function so the backend
is configured in a single place. By default it talks to a locally served
open-weight model through Ollama's OpenAI-compatible API:

    ollama serve
    ollama pull qwen2.5:7b-instruct

Override with environment variables (or .env):
    LLM_BASE_URL  default http://localhost:11434/v1
    LLM_MODEL     default qwen2.5:7b-instruct   (e.g. llama3.2:3b, deepseek-r1:7b)
    LLM_API_KEY   default "ollama" (Ollama ignores it; any hosted
                  OpenAI-compatible server can be used by changing the URL/key)
"""

import os
from typing import Optional, Tuple

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


def build_chat_client() -> Tuple[Optional[OpenAI], str]:
    """Return (client, model_name) for the configured local LLM backend."""
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    api_key = os.environ.get("LLM_API_KEY", "ollama")
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
    except Exception as exc:  # pragma: no cover - construction rarely fails
        print(f"[LLM] Could not create client for {base_url}: {exc}")
        return None, model
    print(f"[LLM] Local backend: {base_url} | model: {model}")
    return client, model

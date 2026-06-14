from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_VLLM_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def is_vllm_configured() -> bool:
    return bool(os.getenv("VLLM_BASE_URL") or os.getenv("USE_VLLM"))


def generate_with_vllm(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.15,
    max_tokens: int = 900,
) -> str:
    endpoint = _chat_endpoint()
    model = os.getenv("VLLM_MODEL", DEFAULT_MODEL)
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _chat_endpoint() -> str:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
    if base_url.endswith("/v1/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return DEFAULT_VLLM_URL if base_url == "http://localhost:8000" else f"{base_url}/v1/chat/completions"


def vllm_base_url() -> str:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
    if base_url.endswith("/v1/chat/completions"):
        return base_url[: -len("/chat/completions")]
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def vllm_model() -> str:
    return os.getenv("VLLM_MODEL", DEFAULT_MODEL)


def vllm_api_key() -> str:
    return os.getenv("VLLM_API_KEY", "EMPTY")

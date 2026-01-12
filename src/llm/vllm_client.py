"""Client for self-hosted vLLM or TGI compatible inference endpoints."""

from __future__ import annotations

import logging
from typing import Iterable, List

import httpx

from config.settings import get_settings
from llm.base import BaseLLMClient

LOGGER = logging.getLogger(__name__)


class VLLMClient(BaseLLMClient):
    """Minimal client for a self-hosted inference server."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.llm_endpoint:
            raise ValueError("Self-hosted LLM endpoint must be configured.")

        self._endpoint = settings.llm_endpoint.rstrip("/")
        self._model = settings.llm_model
        self._api_key = settings.llm_api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def complete(self, prompt: str, *, temperature: float = 0.1) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 512,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self._endpoint}/v1/completions",
                json=payload,
                headers=self._headers(),
            )

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["text"]

    async def chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float = 0.1,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": 768,
        }

        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self._endpoint}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
            )

        response.raise_for_status()
        data = response.json()
        choices: List[dict] = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response contains no choices.")
        return choices[0]["message"]["content"]

"""OpenAI/Azure OpenAI client wrapper."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable

from openai import AsyncOpenAI

from config.settings import get_settings
from llm.base import BaseLLMClient

LOGGER = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Wrapper for OpenAI or Azure OpenAI Chat Completion API."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.llm_api_key:
            raise ValueError("LLM API key must be configured for OpenAI client.")

        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_endpoint or None,
        )
        self._model = settings.llm_model

    async def complete(self, prompt: str, *, temperature: float = 0.1) -> str:
        response = await self._client.completions.create(
            model=self._model,
            prompt=prompt,
            max_tokens=512,
            temperature=temperature,
        )
        return response.choices[0].text

    async def chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float = 0.1,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=768,
        )
        return response.choices[0].message.content or ""

    async def stream_chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=768,
            stream=True,
        )

        async for event in stream:
            try:
                delta = event.choices[0].delta
                chunk = getattr(delta, "content", None)
            except Exception:  # pragma: no cover
                chunk = None
            if chunk:
                yield chunk

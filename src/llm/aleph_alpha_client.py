"""Aleph Alpha Luminous API wrapper."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from aleph_alpha_client import (
    AsyncClient,
    ChatMessage,
    ChatRequest,
    CompletionRequest,
    Prompt,
)

from config.settings import get_settings
from llm.base import BaseLLMClient

LOGGER = logging.getLogger(__name__)


class AlephAlphaClient(BaseLLMClient):
    """Adapter for Aleph Alpha's hosted models (Germany-based)."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.llm_api_key:
            raise ValueError("LLM API key must be configured for Aleph Alpha client.")

        self._client = AsyncClient(token=settings.llm_api_key, host=settings.llm_endpoint)
        self._model = settings.llm_model

    async def complete(self, prompt: str, *, temperature: float = 0.1) -> str:
        request = CompletionRequest(
            prompt=Prompt.from_text(prompt),
            model=self._model,
            temperature=temperature,
            maximum_tokens=512,
        )
        response = await self._client.complete(request)
        return response.completions[0].completion

    async def chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float = 0.1,
    ) -> str:
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages
        ]
        request = ChatRequest(
            model=self._model,
            messages=chat_messages,
            temperature=temperature,
            maximum_output_tokens=768,
        )
        response = await self._client.chat(request)
        return response.messages[-1].content[0].text

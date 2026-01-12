"""Shared abstractions for language model clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Protocol


class LLMResponse(Protocol):
    """Protocol describing the expected response from an LLM call."""

    @property
    def text(self) -> str:  # pragma: no cover - protocol stub
        ...


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, prompt: str, *, temperature: float = 0.1) -> str:
        """Return a completion for the given prompt."""

    @abstractmethod
    async def chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float = 0.1,
    ) -> str:
        """Return a chat-style completion."""

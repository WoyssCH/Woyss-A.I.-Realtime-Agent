"""Factory returning configured LLM client implementation."""

from __future__ import annotations

from config.settings import get_settings
from llm.base import BaseLLMClient
from llm.vllm_client import VLLMClient

try:  # optional imports
    from llm.openai_client import OpenAIClient  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAIClient = None  # type: ignore

try:
    from llm.aleph_alpha_client import AlephAlphaClient  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AlephAlphaClient = None  # type: ignore


def build_llm_client() -> BaseLLMClient:
    """Instantiate the configured LLM connector."""

    settings = get_settings()
    if settings.llm_provider == "self_hosted_vllm":
        return VLLMClient()
    if settings.llm_provider == "openai":
        if OpenAIClient is None:
            raise ImportError("openai package not installed.")
        return OpenAIClient()
    if settings.llm_provider == "aleph_alpha":
        if AlephAlphaClient is None:
            raise ImportError("aleph-alpha-client package not installed.")
        return AlephAlphaClient()
    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")

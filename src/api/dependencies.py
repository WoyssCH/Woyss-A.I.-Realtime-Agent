"""Shared FastAPI dependencies.

Separated to avoid circular imports between route modules.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from agents.assistant_agent import AssistantAgent


@lru_cache(maxsize=1)
def _agent_factory() -> AssistantAgent:
    # Lazy import to avoid importing heavy ML dependencies at module import time.
    from agents.assistant_agent import AssistantAgent

    return AssistantAgent()


def get_agent() -> AssistantAgent:
    return _agent_factory()

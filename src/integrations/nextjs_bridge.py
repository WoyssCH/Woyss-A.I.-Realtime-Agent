"""Bridge for triggering Next.js actions from the assistant."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agents.schemas import ActionDirective
from config.settings import get_settings

LOGGER = logging.getLogger(__name__)


class NextJSBridge:
    """Simple HTTP bridge to Next.js API routes."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.nextjs_action_endpoint:
            raise ValueError("Next.js action endpoint is not configured.")
        self._endpoint = settings.nextjs_action_endpoint.rstrip("/")
        self._api_key = settings.nextjs_action_api_key

    async def dispatch(self, directive: ActionDirective) -> None:
        payload: dict[str, Any] = {
            "action": directive.action,
            "payload": directive.payload,
            "confidence": directive.confidence,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self._endpoint,
                json=payload,
                headers=headers,
            )
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.error("Next.js action dispatch failed: %s", exc)
            raise

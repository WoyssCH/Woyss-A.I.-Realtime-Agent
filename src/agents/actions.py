"""Action planning for integration with external systems."""

from __future__ import annotations

import json
import logging
from typing import Iterable, List

from agents.schemas import ActionDirective, UtteranceInput
from llm.base import BaseLLMClient

LOGGER = logging.getLogger(__name__)

ACTION_PROMPT = """Analysiere das folgende Gespraech aus einer Zahnarztpraxis.
Wenn eine operative Aktion erforderlich ist (z.B. Termin buchen,
Patientendaten aktualisieren, Kostenvoranschlag senden), liefere eine Liste von Aktionen.
Verwende nur Aktionen, die eindeutig begruendet sind. Antworte im JSON-Format:
{
  "actions": [
    {
      "action": "<snake_case_action_name>",
      "payload": { ... },
      "confidence": <float 0-1>
    }
  ]
}
Wenn keine Aktion klar ist, gib eine leere Liste zurueck."""


class ActionPlanner:
    """LLM-backed planner that translates dialogue into system directives."""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm = llm_client

    async def plan(self, utterances: Iterable[UtteranceInput]) -> list[ActionDirective]:
        transcript = self._render_transcript(utterances)
        messages = [
            {"role": "system", "content": ACTION_PROMPT},
            {"role": "user", "content": transcript},
        ]
        raw = await self._llm.chat(messages, temperature=0.0)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("Action planner returned invalid JSON: %s", raw)
            return []

        actions = payload.get("actions", [])
        directives: list[ActionDirective] = []
        for action in actions:
            try:
                directives.append(
                    ActionDirective(
                        action=action["action"],
                        payload=action.get("payload", {}),
                        confidence=float(action.get("confidence", 0.0)),
                    )
                )
            except Exception as exc:
                LOGGER.debug("Skipping invalid action payload %s: %s", action, exc)
        return directives

    def _render_transcript(self, utterances: Iterable[UtteranceInput]) -> str:
        lines: List[str] = []
        for utt in utterances:
            lines.append(f"{utt.speaker}: {utt.text}")
        return "\n".join(lines)

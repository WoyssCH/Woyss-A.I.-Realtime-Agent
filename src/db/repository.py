"""Repository utilities for persisting conversations and facts."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import desc, select
from sqlalchemy.exc import NoResultFound

from db.base import AsyncSessionFactory
from db.models import Conversation, StructuredFact, Utterance


class ConversationRepository:
    """Async repository encapsulating storage operations."""

    async def get_or_create_conversation(self, conversation_id: str) -> Conversation:
        async with AsyncSessionFactory() as session:
            try:
                return await self._get_conversation(session, conversation_id)
            except NoResultFound:
                conversation = Conversation(conversation_id=conversation_id)
                session.add(conversation)
                await session.commit()
                await session.refresh(conversation)
                return conversation

    async def _get_conversation(self, session, conversation_id: str) -> Conversation:
        query = select(Conversation).where(Conversation.conversation_id == conversation_id)
        result = await session.execute(query)
        conversation = result.scalar_one()
        return conversation

    async def add_utterance(
        self,
        conversation_id: str,
        *,
        speaker: str,
        language: str,
        text: str,
        confidence: float,
        attributes: dict,
    ) -> Utterance:
        async with AsyncSessionFactory() as session:
            conversation = await self._get_conversation(session, conversation_id)
            utterance = Utterance(
                conversation_id=conversation.id,
                speaker=speaker,
                language=language,
                text=text,
                confidence=confidence,
                attributes=attributes,
            )
            session.add(utterance)
            await session.commit()
            await session.refresh(utterance)
            return utterance

    async def add_structured_facts(
        self,
        conversation_id: str,
        facts: Sequence[dict],
    ) -> list[StructuredFact]:
        async with AsyncSessionFactory() as session:
            conversation = await self._get_conversation(session, conversation_id)
            persisted: list[StructuredFact] = []
            for fact_payload in facts:
                fact = StructuredFact(
                    conversation_id=conversation.id,
                    source_utterance_id=fact_payload.get("source_utterance_id"),
                    category=fact_payload["category"],
                    field_name=fact_payload["field_name"],
                    value=fact_payload["value"],
                    confidence=fact_payload["confidence"],
                    evidence=fact_payload["evidence"],
                )
                session.add(fact)
                persisted.append(fact)
            await session.commit()
            for fact in persisted:
                await session.refresh(fact)
            return persisted

    async def list_structured_facts(self, conversation_id: str) -> list[StructuredFact]:
        async with AsyncSessionFactory() as session:
            conversation = await self._get_conversation(session, conversation_id)
            query = select(StructuredFact).where(
                StructuredFact.conversation_id == conversation.id
            )
            result = await session.execute(query)
            return list(result.scalars().all())

    async def list_recent_utterances(
        self,
        conversation_id: str,
        *,
        limit: int = 20,
    ) -> list[Utterance]:
        async with AsyncSessionFactory() as session:
            conversation = await self._get_conversation(session, conversation_id)
            query = (
                select(Utterance)
                .where(Utterance.conversation_id == conversation.id)
                .order_by(desc(Utterance.started_at), desc(Utterance.id))
                .limit(limit)
            )
            result = await session.execute(query)
            # Return oldest -> newest for LLM context.
            return list(reversed(list(result.scalars().all())))

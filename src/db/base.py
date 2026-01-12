"""Database engine and base declarative models."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base model."""


settings = get_settings()
engine = create_async_engine(settings.database_url, echo=False, future=True)
AsyncSessionFactory = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def init_db() -> None:
    """Initialize database schema.

    In local/dev environments we can auto-create tables. In higher environments,
    prefer Alembic migrations and set `AUTO_CREATE_DB_SCHEMA=false`.
    """

    if not settings.auto_create_db_schema:
        return

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

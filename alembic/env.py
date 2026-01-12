from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# Alembic Config object, provides access to values in alembic.ini
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Ensure `src/` is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from config.settings import get_settings  # noqa: E402
from db.base import Base  # noqa: E402

# Import models so metadata is populated.
import db.models  # noqa: F401,E402

target_metadata = Base.metadata


def _get_database_url() -> str:
    # Allow override for CI/tests.
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""

    url = _get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""

    connectable: AsyncEngine = create_async_engine(
        _get_database_url(),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

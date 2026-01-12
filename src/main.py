"""Entry point for the Swiss dental practice voice assistant service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router as api_router
from config.settings import get_settings
from db.base import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Swiss Dental Virtual Assistant",
    description="Speech-driven AI assistant for Swiss dental practices.",
    lifespan=lifespan,
)
app.include_router(api_router, prefix="/api")

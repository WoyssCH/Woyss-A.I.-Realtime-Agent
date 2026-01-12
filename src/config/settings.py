"""Application-wide configuration loading and validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    environment: Literal["local", "dev", "prod"] = Field(default="local")
    log_level: str = Field(default="INFO")

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/assistant.db",
        description="SQLAlchemy connection string.",
    )

    # Migrations / schema
    auto_create_db_schema: bool = Field(
        default=True,
        description="If true, creates tables automatically on startup (useful for local/dev).",
    )

    # Speech recognition
    whisper_model_size: str = Field(default="Systran/faster-whisper-large-v3")
    whisper_compute_type: str = Field(default="auto")  # e.g. float16, int8_float16
    whisper_device: str = Field(default="auto")

    # Text to speech
    tts_provider: Literal["azure", "coqui"] = Field(default="azure")
    azure_speech_key: str | None = Field(default=None)
    azure_speech_region: str | None = Field(default=None)
    default_voice_female: str = Field(default="de-CH-LeniNeural")

    # LLM connectivity
    llm_provider: Literal["self_hosted_vllm", "openai", "aleph_alpha"] = Field(
        default="self_hosted_vllm"
    )
    llm_endpoint: str | None = Field(
        default=None, description="HTTP endpoint for the self-hosted inference server."
    )
    llm_api_key: str | None = Field(default=None)
    llm_model: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Model identifier for the selected provider.",
    )

    # Information extraction
    extraction_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # Action bridge
    nextjs_action_endpoint: str | None = Field(
        default=None,
        description="Optional Next.js action endpoint to notify about structured events.",
    )
    nextjs_action_api_key: str | None = Field(default=None)

    data_dir: Path = Field(default=Path("./data"))

    @field_validator("data_dir")
    @classmethod
    def ensure_data_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()

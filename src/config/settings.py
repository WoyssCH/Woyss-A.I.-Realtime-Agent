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

    # Twilio (Voice)
    twilio_account_sid: str | None = Field(default=None)
    twilio_auth_token: str | None = Field(default=None)
    twilio_from_number: str | None = Field(default=None, description="E.164, e.g. +4144...")
    public_base_url: str | None = Field(
        default=None,
        description="Public base URL for Twilio webhooks (e.g. https://<ngrok>.ngrok-free.app).",
    )
    twilio_say_language: str = Field(default="de-CH")
    twilio_enable_media_streams: bool = Field(
        default=False,
        description="If true, exposes experimental Twilio Media Streams endpoints.",
    )
    twilio_stream_pause_seconds: int = Field(
        default=6,
        description="How long Twilio should stream before redirecting to speak a response.",
    )
    twilio_recall_api_key: str | None = Field(
        default=None,
        description="Optional API key required to call the recall endpoint.",
    )

    # Twilio SIP / Asterisk true-duplex path
    twilio_sip_target_uri: str | None = Field(
        default=None,
        description=(
            "Optional SIP URI for Twilio <Dial><Sip>. "
            "Example: sip:assistant@your-domain.sip.twilio.com"
        ),
    )

    # Asterisk ARI controller (for bridging call audio to the external media server)
    asterisk_ari_url: str = Field(
        default="http://asterisk:8088/ari",
        description="Base URL for Asterisk ARI, e.g. http://localhost:8088/ari",
    )
    asterisk_ari_username: str | None = Field(default=None)
    asterisk_ari_password: str | None = Field(default=None)
    asterisk_stasis_app: str = Field(
        default="woyss",
        description="ARI stasis application name used by the dialplan.",
    )

    # External media server UDP endpoint (Asterisk will send RTP here)
    media_server_host: str = Field(default="media", description="Host/IP where the Python media server listens.")
    media_server_port: int = Field(default=20000, description="UDP port for RTP (G.711 mu-law).")
    media_rtp_payload_type: int = Field(
        default=0,
        description="RTP payload type for PCMU (G.711 mu-law). Typically 0.",
    )

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

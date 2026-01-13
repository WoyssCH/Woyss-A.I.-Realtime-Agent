from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_settings


@dataclass(frozen=True)
class TwilioConfig:
    account_sid: str
    auth_token: str
    from_number: str
    public_base_url: str


def get_twilio_config() -> TwilioConfig:
    settings = get_settings()
    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        raise ValueError("Twilio credentials are not configured")
    if not settings.twilio_from_number:
        raise ValueError("Twilio from-number is not configured")
    if not settings.public_base_url:
        raise ValueError("PUBLIC_BASE_URL is required for Twilio callbacks")

    return TwilioConfig(
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        from_number=settings.twilio_from_number,
        public_base_url=settings.public_base_url.rstrip("/"),
    )


def build_twilio_client():
    from twilio.rest import Client

    cfg = get_twilio_config()
    return Client(cfg.account_sid, cfg.auth_token)

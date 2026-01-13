from __future__ import annotations

from fastapi.testclient import TestClient

from integrations.twilio_client import TwilioConfig


class FakeAgent:
    async def handle_text(
        self,
        conversation_id: str,
        *,
        speaker: str,
        text: str,
        language: str = "de",
        confidence: float = 1.0,
        attributes=None,
    ):
        return {
            "transcript": text,
            "assistant_response": "Alles klar. Was ist Ihr Anliegen?",
            "language": language,
            "confidence": confidence,
            "structured_facts": [],
            "actions": [],
        }

    async def start_conversation(self, conversation_id: str) -> None:
        return None


class FakeTwilioCall:
    def __init__(self, sid: str) -> None:
        self.sid = sid


class FakeTwilioCalls:
    def create(self, *, to: str, from_: str, url: str, method: str):
        assert to
        assert from_
        assert url
        assert method == "POST"
        return FakeTwilioCall("CA123")


class FakeTwilioClient:
    def __init__(self) -> None:
        self.calls = FakeTwilioCalls()


def test_twilio_voice_webhook_returns_gather_when_no_speech(app, monkeypatch):
    import api.dependencies as deps

    app.dependency_overrides[deps.get_agent] = lambda: FakeAgent()

    with TestClient(app) as client:
        resp = client.post("/api/twilio/voice", data={"CallSid": "CA111"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")
    assert "<Gather" in resp.text


def test_twilio_voice_webhook_says_response_when_speech_present(app, monkeypatch):
    import api.dependencies as deps

    app.dependency_overrides[deps.get_agent] = lambda: FakeAgent()

    with TestClient(app) as client:
        resp = client.post(
            "/api/twilio/voice",
            data={"CallSid": "CA111", "SpeechResult": "Hallo"},
        )

    assert resp.status_code == 200
    assert "<Say" in resp.text
    assert "Alles klar" in resp.text


def test_twilio_recall_creates_call(app, monkeypatch):
    import api.twilio_routes as twilio_routes

    app.dependency_overrides[twilio_routes.get_twilio_client] = lambda: FakeTwilioClient()
    app.dependency_overrides[twilio_routes.get_twilio_cfg] = (
        lambda: TwilioConfig(
            account_sid="AC123",
            auth_token="token",
            from_number="+15005550006",
            public_base_url="https://example.com",
        )
    )

    with TestClient(app) as client:
        resp = client.post("/api/twilio/recalls", json={"to_number": "+41791234567"})

    assert resp.status_code == 200
    assert resp.json()["call_sid"] == "CA123"


def test_twilio_voice_stream_returns_404_when_disabled(app):
    with TestClient(app) as client:
        resp = client.post("/api/twilio/voice_stream", data={"CallSid": "CA111"})
    assert resp.status_code == 404

from __future__ import annotations

from fastapi.testclient import TestClient

from agents.errors import NoSpeechDetectedError


class ErroringAgent:
    async def start_conversation(self, conversation_id: str) -> None:
        return None

    async def handle_audio(self, conversation_id: str, speaker: str, audio_bytes: bytes) -> dict:
        raise NoSpeechDetectedError()


def test_upload_audio_maps_assistant_errors_to_http(app):
    import api.routes as routes

    app.dependency_overrides[routes.get_agent] = lambda: ErroringAgent()

    with TestClient(app) as client:
        response = client.post(
            "/api/conversations/demo/audio?speaker=patient",
            files={"audio_file": ("sample.wav", b"RIFF....WAVE", "audio/wav")},
        )

    assert response.status_code == 422
    assert "Keine Spracheingabe" in response.json()["detail"]

from __future__ import annotations

import asyncio


def test_start_conversation(client):
    response = client.post("/api/conversations/demo/start")
    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "demo"
    assert payload["status"] == "started"


def test_upload_audio_missing_file_returns_422(client):
    response = client.post("/api/conversations/demo/audio?speaker=patient")
    assert response.status_code == 422


def test_upload_audio_rejects_unsupported_content_type(client):
    response = client.post(
        "/api/conversations/demo/audio?speaker=patient",
        files={"audio_file": ("sample.txt", b"not-a-wav", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported audio format."


def test_upload_audio_success_returns_expected_shape(client):
    response = client.post(
        "/api/conversations/demo/audio?speaker=patient",
        files={"audio_file": ("sample.wav", b"RIFF....WAVE", "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["transcript"]
    assert payload["assistant_response"]
    assert payload["assistant_audio_b64"]
    assert payload["assistant_audio_mime"]
    assert isinstance(payload["structured_facts"], list)
    assert isinstance(payload["actions"], list)


def test_get_facts_returns_persisted_facts(client):
    # Prepopulate DB via repository (async), then validate API response.
    async def _seed():
        from db.base import init_db
        from db.repository import ConversationRepository

        await init_db()
        repo = ConversationRepository()
        await repo.get_or_create_conversation("facts-demo")
        await repo.add_structured_facts(
            "facts-demo",
            [
                {
                    "category": "patient",
                    "field_name": "patient_name",
                    "value": "Max Muster",
                    "confidence": 0.9,
                    "evidence": "Ich heisse Max Muster.",
                }
            ],
        )

    asyncio.run(_seed())

    response = client.get("/api/conversations/facts-demo/facts")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["field_name"] == "patient_name"
    assert payload[0]["value"] == "Max Muster"

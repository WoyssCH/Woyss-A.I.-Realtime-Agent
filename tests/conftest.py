from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class FakeAgent:
    async def start_conversation(self, conversation_id: str) -> None:
        return None

    async def handle_audio(self, conversation_id: str, speaker: str, audio_bytes: bytes) -> dict:
        return {
            "transcript": "Hallo",
            "assistant_response": "Guten Tag! Wie kann ich helfen?",
            "language": "de",
            "confidence": 0.99,
            "assistant_audio": b"FAKEAUDIO",
            "assistant_audio_mime": "audio/mp3",
            "structured_facts": [],
            "actions": [],
        }


@pytest.fixture(scope="session")
def app(tmp_path_factory: pytest.TempPathFactory):
    tmp_dir = tmp_path_factory.mktemp("runtime")
    db_path = tmp_dir / "assistant_test.db"

    # Must be set before importing modules that create the SQLAlchemy engine.
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    os.environ["DATA_DIR"] = str(tmp_dir)
    # Ensure tests can rely on the schema existing without running Alembic.
    os.environ["AUTO_CREATE_DB_SCHEMA"] = "true"

    import importlib

    # Ensure clean import with the test DB settings.
    for module_name in [
        "config.settings",
        "db.base",
        "db.models",
        "db.repository",
        "api.routes",
        "main",
    ]:
        sys.modules.pop(module_name, None)

    main = importlib.import_module("main")
    return main.app


@pytest.fixture()
def client(app):
    # Override agent dependency so tests never instantiate Whisper/TTS/LLM.
    import api.dependencies as deps

    app.dependency_overrides[deps.get_agent] = lambda: FakeAgent()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()

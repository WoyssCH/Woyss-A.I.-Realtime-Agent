# Swiss Dental Voice Assistant

Python-based real-time voice agent tailored for Swiss dental practices. The assistant captures multilingual speech, extracts business-critical facts with verification, persists them in a database, and can trigger external actions (e.g. Next.js API routes). The voice persona is a natural, friendly female praxis assistant who handles Swiss German, High German, French, Italian, and Romansh.

## Highlights
- **Speech stack**: Offline Whisper (`faster-whisper`) for high-accuracy transcription across the supported Swiss languages.  
- **Verified information capture**: Dual-pass LLM workflow (extraction + verification) ensures only high-confidence facts enter the database.  
- **Natural female voice**: Choose Azure Cognitive Services (Swiss region) or offline Coqui TTS.  
- **Self-hosted LLM ready**: Default connector targets a vLLM/TGI endpoint so you can host the model within Swiss infrastructure. OpenAI or Aleph Alpha connectors are included as alternatives.  
- **Action bridge**: Optional HTTP bridge to invoke external systems such as a Next.js application.  
- **Persistence**: Async SQLAlchemy models for conversations, utterances, and structured facts.

## Project layout
```
src/
  config/            # Pydantic-based settings loader
  speech/            # Whisper transcription + TTS synthesizers
  llm/               # LLM client abstractions and providers
  agents/            # Assistant orchestration, extraction, action planning
  db/                # SQLAlchemy models and repositories
  api/               # FastAPI routes and schemas
  integrations/      # Optional bridges (Next.js)
  main.py            # FastAPI application entry point
requirements.txt     # Python dependencies
.env.example         # Configuration template
```

## Prerequisites
- Python 3.10 or newer
- FFmpeg installed when using Coqui TTS (required by the TTS library)
- Whisper models are downloaded automatically on first run
- For Azure speech: Cognitive Services Speech resource in `switzerlandnorth` (or `switzerlandwest`) with Neural female voices such as `de-CH-LeniNeural`
- For self-hosted LLM: vLLM or Text Generation Inference server exposing OpenAI-compatible `/v1/chat/completions`

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust credentials:

- `TTS_PROVIDER=azure` plus `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION=switzerlandnorth`
- `LLM_PROVIDER=self_hosted_vllm` with `LLM_ENDPOINT` pointing to your self-hosted model inside Switzerland
- Adjust `LLM_MODEL` to the deployed instruct model (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
- Leave `NEXTJS_ACTION_ENDPOINT` empty if you do not need outbound actions

## Running the service
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
```

Endpoints:
- `POST /api/conversations/{conversation_id}/start` - initialise a session
- `POST /api/conversations/{conversation_id}/audio?speaker=patient` - upload `audio/wav` (16 kHz mono recommended). Returns transcript, assistant reply text, base64 audio (MP3/WAV bytes), structured facts, and planned actions.
- `GET /api/conversations/{conversation_id}/facts` - retrieve persisted structured facts

Use multipart form submissions for audio uploads:
```bash
curl -X POST "http://localhost:8080/api/conversations/demo/audio?speaker=patient" ^
  -F "audio_file=@sample.wav"
```

## Recommended Hugging Face models
- Speech-to-text: `Systran/faster-whisper-large-v3` (default in `.env.example`)
- Text generation: `meta-llama/Meta-Llama-3-8B-Instruct` (serve via vLLM/TGI)
- Text-to-speech: `coqui/XTTS-v2` (female Swiss voices via speaker cloning)

Use `huggingface-cli download` to prefetch the models and mount the cache inside the container.

## Deployment
- Build with the provided `Dockerfile` and `docker-compose.yaml` (GPU reservations included).
- Copy `.env.example` to `.env`, update credentials, then run `docker compose up -d`.
- Detailed guidance (model caching, server sizing, observability) lives in `docs/deployment.md`.

## Branching & deployment workflow
This repository uses environment-specific branches. Merges/pushes to each branch trigger GitHub Actions that automatically build and deploy to the matching environment.

- `main` -> default showcase branch (stable demo/reference branch)
- `feature` -> feature environment deployment (new features land here first)
- `testing` -> testing/staging environment deployment
- `developement` -> development environment deployment
- `production` -> production environment deployment

Notes:
- `main` should remain runnable and presentable (used for showcasing and as the default branch).
- Keep PRs small and merge forward (e.g. `feature` -> `testing` -> `developement` -> `production`) so each environment receives the same changes in order.
- Each branch includes (or is expected to include) the GitHub Actions workflow files required to build the Docker image and deploy to its environment.

## Data versioning (DVC)
This repo is DVC-initialized for versioning large data artifacts (datasets, sample audio, model files) without bloating Git.

Common commands:
- Track a folder/file: `dvc add data/my_dataset/`
- Push tracked data to your configured remote: `dvc push`
- Pull tracked data (e.g. on a server/CI runner): `dvc pull`

Configure a default remote (example):
```bash
dvc remote add -d origin <REMOTE_URL>
dvc remote modify origin --local <REMOTE_AUTH_KEY> <REMOTE_AUTH_VALUE>
```

Notes:
- Store credentials in GitHub Actions secrets (or `.dvc/config.local`) rather than committing them.
- If you deploy from CI, ensure the workflow runs `dvc pull` before building/running if your image/runtime depends on tracked artifacts.

## Database migrations (Alembic)
This project uses Alembic for database migrations.

Common commands:
- Create a new revision (after changing models): `alembic revision -m "describe change" --autogenerate`
- Apply migrations: `alembic upgrade head`
- Roll back one migration: `alembic downgrade -1`

Notes:
- In production-like environments, set `AUTO_CREATE_DB_SCHEMA=false` to avoid `create_all()` and rely on migrations.

## Accuracy safeguards
- Whisper with beam search and contextual prompts for Swiss dental vocabulary
- LLM extraction prompt forces JSON output and labels uncertain facts as rejects
- Second verification call checks every candidate fact and drops anything below `EXTRACTION_CONFIDENCE_THRESHOLD`
- Facts are deduplicated before being written to the database
- Optional action planning only dispatches directives above 0.75 confidence

## Integrating with a Next.js backend
Set `NEXTJS_ACTION_ENDPOINT` to your Next.js API route (HTTPS) and `NEXTJS_ACTION_API_KEY` if your endpoint expects bearer authentication. The assistant will POST high-confidence directives that look like:
```json
{
  "action": "book_follow_up",
  "payload": {
    "patient_name": "Max Muster",
    "requested_date": "2025-01-12"
  },
  "confidence": 0.82
}
```

## Customising voices and languages
- Azure voices:
  - `de-CH-LeniNeural` (Swiss German, female)
  - `fr-CH-ArianeNeural`
  - `it-CH-GiannaNeural`
  - Romansh currently falls back to the Swiss German voice
- For offline deployments, switch to `TTS_PROVIDER=coqui` to use the multilingual XTTS model. Provide a reference speaker wav file if you need a specific timbre.

## Database notes
- Default SQLite database lives in `./data/assistant.db`
- Swap `DATABASE_URL` to PostgreSQL (e.g. `postgresql+asyncpg://user:pass@host/db`) for production
- `db/base.py` exposes `init_db()` which runs automatically on FastAPI startup

## Production hardening checklist
1. Deploy the vLLM server inside a Swiss data centre (e.g. Swiss cloud provider or on-premises) with HTTPS and authentication.
2. Enable TLS termination for the FastAPI service (handled by reverse proxy or API gateway).
3. Add authentication/authorization to the API (FastAPI dependencies).
4. Instrument with structured logging and tracing (e.g. OpenTelemetry).
5. Configure automated tests covering extraction prompts with representative transcripts.
6. Run background jobs to reconcile structured facts against EHR/CRM systems.

## Next steps
- Build a small React dashboard that streams live transcripts and fact updates over WebSocket.
- Add diarisation to differentiate multiple staff members if required.
- Extend the verification stage with retrieval of prior patient records for cross-checking.

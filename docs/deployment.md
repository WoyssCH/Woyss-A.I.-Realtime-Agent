# Deployment Guide

This document summarises the steps required to run the Swiss dental voice assistant in production, recommended Hugging Face models, and minimum server specifications for a realtime experience.

## 1. Container images

The repository includes:

- `Dockerfile` - builds the FastAPI application with `uvicorn`.
- `docker-compose.yaml` - example runtime configuration exposing the API on port `8080` and requesting GPU access (Docker Compose v2 syntax).

Adjust environment variables by copying `.env.example` to `.env` before building.

### Build and run

```bash
docker compose build
docker compose up -d
```

The container expects the writable directory `./data` (mounted automatically) for SQLite storage and any temporary audio files.

## 2. Recommended Hugging Face models

| Purpose          | Model ID                                | Notes                                                                 |
|------------------|-----------------------------------------|-----------------------------------------------------------------------|
| Speech-to-text   | `Systran/faster-whisper-large-v3`       | High accuracy multilingual Whisper variant, robust for Swiss dialects.|
| Text generation  | `meta-llama/Meta-Llama-3-8B-Instruct`   | Strong multilingual instruct model; run via vLLM or TGI.              |
| Text-to-speech   | `coqui/XTTS-v2`                         | Multilingual TTS with speaker cloning; provides natural female voices.|

### Download helpers

Use the Hugging Face CLI for caching (recommended during deployment):

```bash
huggingface-cli download Systran/faster-whisper-large-v3 --local-dir ./models/whisper
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./models/llama3
huggingface-cli download coqui/XTTS-v2 --local-dir ./models/xtts
```

Expose the cache to the container by mounting a volume, e.g. `- ./models:/app/models` and set `HF_HOME=/app/models`.

## 3. Server requirements

For realtime call-centre quality (sub-second response):

- **GPU**: 1x NVIDIA L40S (48 GB) or A100 (40 GB). INT4 quantised Llama 3 and Whisper run comfortably on 40 GB.  
- **CPU**: >= 16 vCPU (AMD EPYC or Intel Xeon) for VAD, preprocessing, and concurrent requests.  
- **RAM**: >= 64 GB system memory.  
- **Disk**: SSD, >= 200 GB for model caches and logs.  
- **Network**: Low latency (<= 20 ms) between API and practice clients; ensure TLS termination.  
- **Driver stack**: CUDA 12.x with cuDNN 8, Docker 24+, NVIDIA Container Toolkit.

For non-realtime backoffice workflows you can downscale to an NVIDIA RTX 4090/RTX 6000 Ada or an A10G with 24 GB VRAM; expect higher transcription latency.

## 4. External services

- **LLM serving**: Run vLLM or Text Generation Inference using the recommended model. Expose an OpenAI-compatible `/v1/chat/completions` endpoint and set `LLM_ENDPOINT` accordingly.  
- **Next.js bridge** (optional): Provide a secured HTTPS endpoint and configure `NEXTJS_ACTION_ENDPOINT`.

## 5. Health checks and observability

Add the following best practices after deployment:

- Liveness endpoint: `GET /api/conversations/health` (implement a thin route if required) for container orchestrators.
- Metrics: integrate Prometheus (`prometheus_client`) and export GPU/latency metrics.
- Logging: direct `uvicorn` access logs to STDOUT and aggregate via your platform (e.g. Loki, Elasticsearch).

## 6. Rolling updates

When updating:

1. Build a new image and push to the registry.
2. Rotate containers one by one (zero downtime) with readiness probes waiting for `uvicorn` to start.
3. Run smoke tests: start a conversation, upload a sample WAV, verify structured facts and action dispatch.

This setup delivers an OpenAI Realtime-like experience while keeping all workloads within your managed GPU infrastructure.

# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src ./src
COPY README.md ./
COPY .env.example ./

EXPOSE 8080

ENV WORKERS=2 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]

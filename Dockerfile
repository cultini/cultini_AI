# syntax=docker/dockerfile:1
FROM python:3.12-slim

# uv: fast, reproducible installs from uv.lock.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps first (cached unless pyproject.toml / uv.lock change).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download nltk data so the first request doesn't pay for it. LlamaIndex
# core can lazily fetch these tokenizers; baking them in keeps cold starts fast.
ENV NLTK_DATA=/usr/share/nltk_data
RUN uv run python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab

# App code (the corpus ships in the image; it's the real source of truth).
COPY . .

# Render injects $PORT; default to 8000 for local `docker run`.
ENV PORT=8000

# Single worker, NO --reload: the RAG index is lru_cache'd per process.
CMD uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1

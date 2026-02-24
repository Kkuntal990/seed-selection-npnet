FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip curl git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ---- Layer 1: pyproject.toml only (cached unless deps change) ----

COPY seed-mining/pyproject.toml seed-mining/pyproject.toml
COPY vlm-eval/pyproject.toml vlm-eval/pyproject.toml
COPY npnet/pyproject.toml npnet/pyproject.toml

# Stub src dirs so uv sync can resolve the local path deps
RUN mkdir -p seed-mining/src/seed_mining && touch seed-mining/src/seed_mining/__init__.py \
    && mkdir -p vlm-eval/src/vlm_eval && touch vlm-eval/src/vlm_eval/__init__.py \
    && mkdir -p npnet/src/npnet && touch npnet/src/npnet/__init__.py

# Install all deps (this layer is cached until a pyproject.toml changes)
WORKDIR /app/seed-mining
RUN uv sync --no-dev
WORKDIR /app/vlm-eval
RUN uv sync --no-dev
WORKDIR /app/npnet
RUN uv sync --no-dev

# ---- Layer 2: actual source code (changes frequently, but deps are cached) ----

WORKDIR /app

COPY seed-mining/src/ seed-mining/src/
COPY seed-mining/prompt_dataset/ seed-mining/prompt_dataset/
COPY seed-mining/scripts/ seed-mining/scripts/

COPY vlm-eval/src/ vlm-eval/src/
COPY vlm-eval/scripts/ vlm-eval/scripts/

COPY npnet/src/ npnet/src/
COPY npnet/scripts/ npnet/scripts/

# Re-install in editable-like mode so the new source is picked up
WORKDIR /app/seed-mining
RUN uv sync --no-dev
WORKDIR /app/vlm-eval
RUN uv sync --no-dev
WORKDIR /app/npnet
RUN uv sync --no-dev

WORKDIR /app

CMD ["/bin/bash"]

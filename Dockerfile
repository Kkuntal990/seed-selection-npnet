FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip curl git \
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

# Stub src dirs so uv can resolve the local editable packages
RUN mkdir -p seed-mining/src/seed_mining && touch seed-mining/src/seed_mining/__init__.py \
    && mkdir -p vlm-eval/src/vlm_eval && touch vlm-eval/src/vlm_eval/__init__.py \
    && mkdir -p npnet/src/npnet && touch npnet/src/npnet/__init__.py

# Install all deps into system Python (cached until a pyproject.toml changes)
# Only install vlm-eval and npnet — they pull in seed-mining as a path dependency
RUN uv pip install --system -e ./vlm-eval -e ./npnet && \
    uv pip install --system jupyterlab

# ---- Layer 2: actual source code (changes frequently, but deps are cached) ----

COPY seed-mining/src/ seed-mining/src/
COPY seed-mining/prompt_dataset/ seed-mining/prompt_dataset/
COPY seed-mining/scripts/ seed-mining/scripts/

COPY vlm-eval/src/ vlm-eval/src/
COPY vlm-eval/scripts/ vlm-eval/scripts/

COPY npnet/src/ npnet/src/
COPY npnet/scripts/ npnet/scripts/

# Re-install editable packages so new source is picked up (no-deps = fast, deps cached)
RUN uv pip install --system --no-deps -e ./seed-mining -e ./vlm-eval -e ./npnet

WORKDIR /app

CMD ["/bin/bash"]

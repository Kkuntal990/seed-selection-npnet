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

# --- Copy all subprojects ---

# seed-mining (base dependency for vlm-eval and npnet)
COPY seed-mining/pyproject.toml seed-mining/pyproject.toml
COPY seed-mining/src/ seed-mining/src/
COPY seed-mining/prompt_dataset/ seed-mining/prompt_dataset/
COPY seed-mining/scripts/ seed-mining/scripts/

# vlm-eval
COPY vlm-eval/pyproject.toml vlm-eval/pyproject.toml
COPY vlm-eval/src/ vlm-eval/src/
COPY vlm-eval/scripts/ vlm-eval/scripts/

# npnet
COPY npnet/pyproject.toml npnet/pyproject.toml
COPY npnet/src/ npnet/src/
COPY npnet/scripts/ npnet/scripts/

# --- Install each project ---

# seed-mining
WORKDIR /app/seed-mining
RUN uv sync --no-dev

# vlm-eval (depends on seed-mining via path)
WORKDIR /app/vlm-eval
RUN uv sync --no-dev

# npnet (depends on seed-mining via path)
WORKDIR /app/npnet
RUN uv sync --no-dev

WORKDIR /app

# No fixed entrypoint — use as general-purpose image for jobs or Jupyter
CMD ["/bin/bash"]

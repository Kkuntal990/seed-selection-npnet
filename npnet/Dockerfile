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

# Copy seed-mining (path dependency)
COPY seed-mining/pyproject.toml seed-mining/pyproject.toml
COPY seed-mining/src/ seed-mining/src/
COPY seed-mining/prompt_dataset/ seed-mining/prompt_dataset/

# Copy npnet project
COPY npnet/pyproject.toml npnet/pyproject.toml
COPY npnet/src/ npnet/src/
COPY npnet/scripts/ npnet/scripts/

# Install from npnet directory
WORKDIR /app/npnet
RUN uv sync --no-dev

# Default entrypoint
ENTRYPOINT ["uv", "run", "python"]
CMD ["scripts/collect_noise_pairs.py", "--dry_run"]

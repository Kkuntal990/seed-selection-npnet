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

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY prompt_dataset/ prompt_dataset/
COPY scripts/ scripts/

# Install dependencies (no dev, no xformers by default)
RUN uv sync --no-dev

# Default entrypoint
ENTRYPOINT ["uv", "run", "python", "scripts/generate_seed_mining_images.py"]

# Default args (override in K8s job spec)
CMD ["--out_dir", "/output", "--dry_run"]

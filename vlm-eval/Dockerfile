FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Build from repo root: docker build -f vlm-eval/Dockerfile -t vlm-eval .

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

# Copy vlm-eval
COPY vlm-eval/pyproject.toml .
COPY vlm-eval/src/ src/
COPY vlm-eval/scripts/ scripts/

# Install dependencies
RUN uv sync --no-dev

# Default entrypoint
ENTRYPOINT ["uv", "run", "python", "scripts/run_evaluation.py"]

CMD ["--images_dir", "/output/image_gen", "--eval_out_dir", "/output/eval_results", "--seed_range", "2", "--num_objects", "1", "--num_settings", "1"]

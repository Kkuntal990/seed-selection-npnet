# ECE 271B — Seed Selection / NPNet Project

Based on: [All Seeds Are Not Equal (ICLR 2025)](https://arxiv.org/abs/2411.18810)
Official repo: [doub7e/Reliable-Random-Seeds](https://github.com/doub7e/Reliable-Random-Seeds)

## Repository Structure

```
ece271b-project/
  CLAUDE.md                              # This file
  seed-mining/                           # Step 1: Diffusion image generation
    environment.yml                      # Conda env (Python 3.11, pip, git)
    pyproject.toml                       # PEP 621, hatchling, deps via uv
    uv.lock                             # Locked dependency versions
    .python-version                     # 3.11
    .gitignore
    .pre-commit-config.yaml
    ruff.toml
    mypy.ini
    prompt_dataset/                      # FROM official repo (ground truth data)
      objects_train.txt                  # 60 train objects (singular, article, plural)
      objects_eval.txt                   # 30 eval objects
      backgrounds_train.txt             # 8 train backgrounds
      backgrounds_eval.txt              # 4 eval backgrounds
      spatial_prompts_train.txt         # 320 spatial prompts (GPT-4o filtered)
      spatial_prompts_eval.txt          # 160 spatial prompts
    Dockerfile                          # CUDA 12.1 + Python 3.11 + uv
    .dockerignore
    k8s/
      seed-mining-job.yaml              # Single-GPU K8s job
      seed-mining-job-multigpu.yaml     # Multi-GPU K8s job (4x)
      pvcs.yaml                         # PersistentVolumeClaims
    src/seed_mining/
      __init__.py
      config.py                         # SeedMiningConfig (pydantic-settings v2)
      logging_utils.py                  # Logging + rich + throughput tracker
      prompts.py                        # Load prompt data files, build prompt grids
      io_utils.py                       # Atomic writes, resume, paths, metadata
      generator.py                      # Pipeline, batching, OOM fallback, multi-GPU
      cli.py                            # Thin CLI entrypoint
    scripts/
      generate_seed_mining_images.py    # Wrapper calling cli.main()
    tests/
      __init__.py
      test_prompts.py                   # Prompt counts, ordering, stability
      test_paths_and_ids.py             # Path uniqueness, atomic writes, resume
    README.md
```

## What Each File Does

### Config & Environment

- **environment.yml** — Lightweight conda env: Python 3.11 + pip + git only. ML packages come from uv.
- **pyproject.toml** — PEP 621 metadata. Core deps: torch, diffusers>=0.30, transformers<=4.45, accelerate, pydantic-settings, Pillow, rich, tqdm. Dev deps: ruff, mypy, pytest, pre-commit. Build: hatchling with src layout.
- **ruff.toml** — Linter/formatter. Black-compatible, line-length 100, py311.
- **mypy.ini** — Type checker with `ignore_missing_imports = True`.

### Prompt Data (`prompt_dataset/`)

Copied from the official Reliable-Random-Seeds repo. These are the ground truth.

- **objects_{train,eval}.txt** — Format: `"singular, article singular, plural"` per line. Script extracts the plural form for numeracy prompts.
- **backgrounds_{train,eval}.txt** — One background setting per line (e.g., "on a snowy mountain").
- **spatial_prompts_{train,eval}.txt** — Pre-constructed spatial relation prompts (e.g., "A pineapple on the left of a monkey"). Created by random pair generation + GPT-4o filtering for reasonableness.

### Source Modules (`src/seed_mining/`)

- **config.py** — `SeedMiningConfig(BaseSettings)` with CLI args: `--model_id`, `--out_dir`, `--seed_range`, `--split {train,eval,all}`, `--batch_size`, `--generator_device`, `--scheduler`, `--enable_xformers`, `--enable_cpu_offload`, `--dry_run`, etc.
- **prompts.py** — Loads objects/backgrounds/spatial data from text files. Builds deterministic prompt grids:
  - Numeracy: count(2-6) × objects × backgrounds. Template: `"{count_word} {plural}, {background}"`
  - Spatial: pre-constructed prompts, optionally × backgrounds
  - Returns frozen `Prompt` dataclasses with sequential IDs per category
- **io_utils.py** — Atomic image writes (temp+rename), JSONL metadata, resume scanning, path construction, run config serialization, completeness verification.
- **generator.py** — Adapted from official repo. Loads StableDiffusionPipeline (fp16), batched generation with OOM fallback (halve batch + retry), multi-GPU seed sharding via accelerate `PartialState`, resume-safe.
- **logging_utils.py** — Rich console (rank 0) + file handler. `ThroughputTracker` for images/sec and ETA.
- **cli.py** — Instantiates config (auto-parses CLI), calls `run_generation()`.

### Dataset Specification (Comp90)

**Paper defaults (62k images):** `--split train --num_objects 15 --num_settings 4 --append_background_to_spatial false`

| Category | Formula | Per seed | × 100 seeds |
|----------|---------|----------|-------------|
| Numeracy | 5 counts × 15 objects × 4 backgrounds | 300 | 30,000 |
| Spatial | 80 per relation × 4 relations (no bg) | 320 | 32,000 |
| **Total** | | **620** | **62,000** |

**Full Comp90 dataset** (override with `--num_objects null --num_settings null --split all --append_background_to_spatial true`):

| Split | Objects | Backgrounds | Numeracy Prompts | Spatial Prompts |
|-------|---------|-------------|-----------------|-----------------|
| Train | 60 | 8 | 5×60×8 = 2,400 | 320 (×8 bg = 2,560) |
| Eval | 30 | 4 | 5×30×4 = 600 | 160 (×4 bg = 640) |
| All | 90 | 12 | 5,400 | 480 (5,760 w/ bg) |

Numeracy count words: `{2:"two", 3:"three", 4:"four", 5:"five", 6:"six"}`

Spatial relations in data: "on the left of", "on the right of", "on top of/on the top of", "under"

## How to Run Commands

### Environment Setup

```bash
cd seed-mining/
conda env create -f environment.yml
conda activate seed-mining
pip install uv
uv sync --all-extras
```

### Single GPU Generation (paper defaults: 62k images)

```bash
python scripts/generate_seed_mining_images.py \
  --model_id stabilityai/stable-diffusion-2-1 \
  --out_dir /data/seed_mining_sd21 \
  --seed_range 100 \
  --batch_size 16 \
  --generator_device cpu
```

### Multi-GPU Generation

```bash
accelerate launch --num_processes 4 scripts/generate_seed_mining_images.py \
  --model_id stabilityai/stable-diffusion-2-1 \
  --out_dir /data/seed_mining_sd21 \
  --seed_range 100 \
  --batch_size 16 \
  --generator_device cpu
```

### Dry Run

```bash
python scripts/generate_seed_mining_images.py --out_dir /tmp/out --dry_run
```

### Tests, Lint, Types

```bash
pytest tests/ -v
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pre-commit run --all-files
```

## Best Practices

1. **Use `uv sync`** after changing deps in pyproject.toml.
2. **Run `ruff check` and `mypy`** before committing. Pre-commit hooks enforce this.
3. **Type hints on all functions** in src/ — mypy must pass.
4. **CPU generator for determinism** — `--generator_device cpu`. Official repo uses cuda, but CPU is more reproducible per Diffusers docs.
5. **Atomic writes** — all image saves go through `save_image_atomic()`.
6. **Resume-safe** — re-running skips existing images, deduplicates metadata. Just re-run on failure.
7. **Don't modify prompt_dataset/ files** — these are ground truth from the official repo. Changes invalidate all prompt IDs.
8. **Batch size 16** fits A6000 at fp16/512x512. OOM auto-halves.
9. **Prompt IDs are per-category** — numeracy and spatial have separate ID spaces starting at 0.
10. **Count words not digits** — use "two", "three", etc. in prompts, matching the official repo.
11. **transformers<=4.45** — official repo pins this; respect it for compatibility.

## Docker & Kubernetes

### Build Docker image

```bash
cd seed-mining/
docker build -t seed-mining:latest .
```

### Run locally with Docker

```bash
docker run --gpus 1 -v /data/output:/output -v ~/.cache/huggingface:/root/.cache/huggingface \
  seed-mining:latest \
  --out_dir /output --seed_range 100 --batch_size 16 --generator_device cpu
```

### Deploy to Kubernetes

```bash
# Create storage
kubectl apply -f k8s/pvcs.yaml

# Single GPU job
kubectl apply -f k8s/seed-mining-job.yaml

# Multi-GPU job (4x)
kubectl apply -f k8s/seed-mining-job-multigpu.yaml

# Monitor
kubectl logs -f job/seed-mining-sd21
```

K8s manifests are in `seed-mining/k8s/`. Edit the `image:` field to point to your container registry before deploying.

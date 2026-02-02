# Seed Mining: Diffusion Image Generation

Generate diffusion images for seed mining using the Comp90 dataset from
["All Seeds Are Not Equal" (ICLR 2025)](https://arxiv.org/abs/2411.18810).

## Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate seed-mining

# 2. Install uv and sync dependencies
pip install uv
uv sync --all-extras

# 3. (Optional) Install pre-commit hooks
pre-commit install
```

## Usage

### Dry run (verify prompt counts)

```bash
python scripts/generate_seed_mining_images.py --out_dir /tmp/out --dry_run
```

### Single GPU (paper defaults: 62k images)

```bash
python scripts/generate_seed_mining_images.py \
  --model_id stabilityai/stable-diffusion-2-1 \
  --out_dir /data/seed_mining_sd21 \
  --seed_range 100 \
  --height 512 --width 512 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --batch_size 16 \
  --generator_device cpu
```

### Multi-GPU (via accelerate)

```bash
accelerate launch --num_processes 4 scripts/generate_seed_mining_images.py \
  --model_id stabilityai/stable-diffusion-2-1 \
  --out_dir /data/seed_mining_sd21 \
  --seed_range 100 \
  --batch_size 16 \
  --generator_device cpu
```

### Key CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--split` | `train` | `train`, `eval`, or `all` |
| `--seed_range` | `100` | Number of seeds (0..N-1) |
| `--seeds_file` | None | File with one seed per line |
| `--batch_size` | `16` | Images per batch |
| `--generator_device` | `cpu` | `cpu` for determinism, `cuda` for speed |
| `--scheduler` | `ddim` | Scheduler: ddim, dpmsolver++, euler, pndm |
| `--enable_xformers` | False | Use xformers memory-efficient attention |
| `--enable_cpu_offload` | False | Offload model to CPU when not in use |
| `--num_objects` | `15` | Limit objects (None = all) |
| `--num_settings` | `4` | Limit backgrounds (None = all) |
| `--append_background_to_spatial` | `False` | Append backgrounds to spatial prompts |
| `--image_format` | `jpg` | `jpg` or `png` |
| `--dry_run` | False | Print counts without generating |

## Dataset (Comp90)

Prompt data files live in `prompt_dataset/` (from the
[official repo](https://github.com/doub7e/Reliable-Random-Seeds)):

### Paper defaults (62k images)

The default configuration matches the paper: **15 objects, 4 backgrounds, train split, no backgrounds on spatial**.

| Category | Formula | Per seed | × 100 seeds |
| --- | --- | --- | --- |
| Numeracy | 5 counts × 15 objects × 4 backgrounds | 300 | 30,000 |
| Spatial | 80 per relation × 4 relations | 320 | 32,000 |
| **Total** | | **620** | **62,000** |

### Full Comp90 dataset

Override with `--num_objects null --num_settings null --split all --append_background_to_spatial true`:

| Split | Objects | Backgrounds | Numeracy | Spatial (w/ bg) |
| --- | --- | --- | --- | --- |
| Train | 60 | 8 | 2,400 | 2,560 |
| Eval | 30 | 4 | 600 | 640 |
| All | 90 | 12 | 5,400 | 5,760 |

## Output structure

```
out_dir/
  run_config.json
  prompts/
    numeracy_prompts.jsonl
    spatial_prompts.jsonl
  images/
    numeracy/seed=000/prompt=0000.jpg
    spatial/seed=000/prompt=0000.jpg
  metadata/
    numeracy_images.jsonl
    spatial_images.jsonl
  logs/
    generation_rank0.log
  _DONE.txt
```

## Docker

### Build

```bash
docker build -t seed-mining:latest .
```

### Run (single GPU)

```bash
docker run --gpus 1 -v /data/output:/output -v ~/.cache/huggingface:/root/.cache/huggingface \
  seed-mining:latest \
  --model_id stabilityai/stable-diffusion-2-1 \
  --out_dir /output \
  --seed_range 100 \
  --batch_size 16 \
  --generator_device cpu
```

## Kubernetes

### Prerequisites

Create PersistentVolumeClaims for output storage and HuggingFace model cache:

```bash
kubectl apply -f k8s/pvcs.yaml
```

### Single GPU job

```bash
# Edit k8s/seed-mining-job.yaml to set your image registry path
kubectl apply -f k8s/seed-mining-job.yaml
```

### Multi-GPU job (4x GPU)

```bash
kubectl apply -f k8s/seed-mining-job-multigpu.yaml
```

### Monitor

```bash
kubectl logs -f job/seed-mining-sd21
```

## Development

```bash
# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Type check
mypy src/

# Tests
pytest tests/ -v
```

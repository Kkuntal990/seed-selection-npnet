"""Benchmark pipeline: generate golden noise images -> VLM eval -> metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
from seed_mining.io_utils import image_path, save_image_atomic
from seed_mining.logging_utils import ThroughputTracker, setup_logging
from seed_mining.prompts import Prompt, build_all_prompts

from npnet.models.npnet import NPNet

if TYPE_CHECKING:
    from npnet.config import BenchmarkConfig

logger = logging.getLogger("npnet.benchmark")


# ---------------------------------------------------------------------------
# Phase 1: Generate golden-noise images
# ---------------------------------------------------------------------------


def _run_golden_generation(config: BenchmarkConfig) -> None:
    """Generate images using golden noise from a trained NPNet.

    Uses the SDXL generation parameters that match the evaluated images:
    guidance=7.5, steps=30, scheduler=euler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build prompts
    numeracy, spatial = build_all_prompts(
        config.prompt_dataset_dir,
        config.split,
        num_objects=config.num_objects,
        num_settings=config.num_settings,
        append_background_to_spatial=config.append_background_to_spatial,
    )
    categories: list[tuple[str, list[Prompt]]] = [("numeracy", numeracy), ("spatial", spatial)]
    seeds = config.seeds

    total = sum(len(seeds) * len(prompts) for _, prompts in categories)
    tracker = ThroughputTracker(total=total)

    if config.dry_run:
        logger.info("DRY RUN: would generate %d golden noise images", total)
        return

    images_dir = config.out_dir / "golden_images"

    # Load SDXL pipeline
    logger.info("Loading pipeline: %s ...", config.model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    # Use euler scheduler to match actual generation config
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if config.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # Load trained NPNet
    logger.info("Loading NPNet checkpoint: %s ...", config.npnet_checkpoint)
    npnet = NPNet(latent_channels=4, latent_resolution=128, text_embed_dim=2048, text_seq_len=77)
    npnet.load_checkpoint(config.npnet_checkpoint)
    npnet.to(device)
    npnet.eval()

    for category, prompts in categories:
        for seed in seeds:
            for prompt in prompts:
                dest = image_path(
                    images_dir, category, seed, prompt.prompt_id, config.image_format
                )
                if dest.exists():
                    tracker.update(1)
                    continue

                # Generate random noise
                generator = torch.Generator(device=config.generator_device).manual_seed(seed)
                latent = torch.randn(
                    1,
                    4,
                    config.height // 8,
                    config.width // 8,
                    generator=generator,
                    dtype=torch.float16,
                ).to(device)

                # Encode prompt + transform to golden noise
                with torch.no_grad():
                    prompt_embeds, _, _, _ = pipe.encode_prompt(
                        prompt=prompt.text,
                        device=device,
                    )
                    golden_latent = npnet(latent, prompt_embeds).half()

                # Generate image with golden noise
                image = pipe(
                    prompt=prompt.text,
                    latents=golden_latent,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    height=config.height,
                    width=config.width,
                ).images[0]

                save_image_atomic(image, dest)
                tracker.update(1)

            logger.info("Seed %d [%s] done | %s", seed, category, tracker.summary_line())

    logger.info("Golden noise generation complete | %s", tracker.summary_line())


# ---------------------------------------------------------------------------
# Phase 2 & 3: VLM evaluation + ranking (delegate to vlm-eval)
# ---------------------------------------------------------------------------


def _run_vlm_evaluation(config: BenchmarkConfig) -> None:
    """Run VLM evaluation on the generated golden noise images."""
    from vlm_eval.eval_config import EvalConfig
    from vlm_eval.evaluator import run_evaluation

    images_dir = config.out_dir / "golden_images"
    eval_out = config.out_dir / "eval"

    eval_config = EvalConfig(
        vlm_model_id=config.vlm_model_id,
        quantize=config.quantize,
        images_dir=images_dir,
        prompt_dataset_dir=config.prompt_dataset_dir,
        split=config.split,
        num_objects=config.num_objects,
        num_settings=config.num_settings,
        append_background_to_spatial=config.append_background_to_spatial,
        seed_start=config.seed_start,
        seed_range=config.seed_range,
        image_format=config.image_format,
        eval_out_dir=eval_out,
    )
    run_evaluation(eval_config)


def _run_vlm_ranking(config: BenchmarkConfig) -> None:
    """Run seed ranking on the VLM evaluation results."""
    from vlm_eval.eval_config import RankingConfig
    from vlm_eval.ranking import run_ranking

    eval_out = config.out_dir / "eval"
    ranking_out = config.out_dir / "ranking"

    ranking_config = RankingConfig(
        eval_results_dir=eval_out,
        output_dir=ranking_out,
    )
    run_ranking(ranking_config)


# ---------------------------------------------------------------------------
# Phase 4: Compute experiment metrics
# ---------------------------------------------------------------------------


def compute_experiment_metrics(
    ranking_dir: Path,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Compute final experiment metrics from ranking outputs.

    Returns dict with per-category overall accuracy and per-subtask accuracy.
    """
    if categories is None:
        categories = ["numeracy", "spatial"]

    metrics: dict[str, Any] = {}

    for category in categories:
        cat_dir = ranking_dir / category
        if not cat_dir.exists():
            continue

        # Overall accuracy
        ranked_path = cat_dir / "ranked_seeds.json"
        if ranked_path.exists():
            data = json.loads(ranked_path.read_text())
            seeds = data["seeds_ranked"]
            overall_acc = sum(s["accuracy"] for s in seeds) / len(seeds) if seeds else 0.0
            metrics[f"{category}_overall_accuracy"] = overall_acc
            metrics[f"{category}_num_seeds"] = len(seeds)

            # Top-5 and bottom-5 seed accuracies
            metrics[f"{category}_top5_mean"] = (
                sum(s["accuracy"] for s in seeds[:5]) / 5 if len(seeds) >= 5 else 0.0
            )
            metrics[f"{category}_bottom5_mean"] = (
                sum(s["accuracy"] for s in seeds[-5:]) / 5 if len(seeds) >= 5 else 0.0
            )

        # Per-subtask accuracy
        subtask_path = cat_dir / "subtask_seeds.json"
        if subtask_path.exists():
            data = json.loads(subtask_path.read_text())
            for subtask, info in data.get("subtasks", {}).items():
                seeds = info.get("seeds_ranked", [])
                mean_acc = sum(s["accuracy"] for s in seeds) / len(seeds) if seeds else 0.0
                metrics[f"{category}_{subtask}_accuracy"] = mean_acc

    return metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_benchmark(config: BenchmarkConfig) -> None:
    """Orchestrate the full benchmark pipeline."""
    setup_logging(config.out_dir / "logs", rank=0)

    if config.dry_run:
        logger.info("DRY RUN: benchmark pipeline")
        _run_golden_generation(config)
        return

    # Phase 1: Generate golden noise images
    if not config.skip_generation:
        logger.info("=== Phase 1: Golden noise generation ===")
        _run_golden_generation(config)
    else:
        logger.info("Skipping generation (--skip_generation)")

    # Phase 2: VLM evaluation
    if not config.skip_evaluation:
        logger.info("=== Phase 2: VLM evaluation ===")
        _run_vlm_evaluation(config)
    else:
        logger.info("Skipping evaluation (--skip_evaluation)")

    # Phase 3: Ranking
    if not config.skip_ranking:
        logger.info("=== Phase 3: Seed ranking ===")
        _run_vlm_ranking(config)
    else:
        logger.info("Skipping ranking (--skip_ranking)")

    # Phase 4: Compute and report metrics
    logger.info("=== Phase 4: Experiment metrics ===")
    ranking_dir = config.out_dir / "ranking"
    metrics = compute_experiment_metrics(ranking_dir)

    # Save metrics
    metrics_path = config.out_dir / "experiment_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    # Log summary
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info("  %s: %.4f", key, value)
        else:
            logger.info("  %s: %s", key, value)

    logger.info("Benchmark complete. Metrics saved to %s", metrics_path)

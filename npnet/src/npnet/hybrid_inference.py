"""Hybrid inference: baseline, rerank, edit, rerank_edit modes.

Four evaluation modes:
1. baseline: Random seed, standard SDXL (no NPNet)
2. rerank: Sample K seeds, score with SeedScorer, pick best
3. edit: Apply HybridNPNet (NPNet + editor) to transform noise
4. rerank_edit: Rerank K seeds, pick best, then apply HybridNPNet
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
from seed_mining.io_utils import image_path, save_image_atomic
from seed_mining.logging_utils import ThroughputTracker, setup_logging
from seed_mining.prompts import Prompt, build_all_prompts

from npnet.models.hybrid_npnet import HybridNPNet
from npnet.models.seed_scorer import SeedScorer

if TYPE_CHECKING:
    from npnet.config import HybridInferenceConfig

logger = logging.getLogger("npnet.hybrid_inference")


def _generate_noise_for_seed(
    seed: int,
    height: int,
    width: int,
    device: torch.device | str,
    generator_device: str = "cpu",
) -> torch.Tensor:
    """Generate a single noise latent for a given seed."""
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    return torch.randn(
        1, 4, height // 8, width // 8,
        generator=generator,
        dtype=torch.float16,
    ).to(device)


def _rerank_seeds(
    scorer: SeedScorer,
    seeds: list[int],
    prompt_embeds: torch.Tensor,
    height: int,
    width: int,
    device: torch.device | str,
    generator_device: str = "cpu",
) -> int:
    """Score multiple seeds and return the best one.

    Args:
        scorer: Trained SeedScorer.
        seeds: Candidate seeds to evaluate.
        prompt_embeds: ``(1, 77, 2048)`` text embeddings.
        height: Image height.
        width: Image width.
        device: Computation device.
        generator_device: Device for RNG.

    Returns:
        Best seed according to scorer.
    """
    # Batch all candidate noises
    noises = torch.cat([
        _generate_noise_for_seed(s, height, width, device, generator_device)
        for s in seeds
    ], dim=0)  # (K, 4, H/8, W/8)

    # Expand prompt_embeds to match batch
    embeds = prompt_embeds.expand(len(seeds), -1, -1)  # (K, 77, 2048)

    with torch.no_grad():
        scores = scorer(noises, embeds)  # (K, 1)

    best_idx = scores.squeeze(-1).argmax().item()
    return seeds[best_idx]


def run_hybrid_evaluation(config: HybridInferenceConfig) -> None:
    """Run hybrid inference evaluation with the specified mode."""
    log_dir = config.out_dir / "logs"
    setup_logging(log_dir, rank=0)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build prompts
    logger.info("Building prompts (split=%s) ...", config.split)
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
        logger.info("DRY RUN: would generate %d images in mode=%s", total, config.hybrid_mode)
        return

    images_dir = config.out_dir / "images"

    # Load SDXL pipeline
    logger.info("Loading SDXL pipeline: %s ...", config.model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if config.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # Load models based on mode
    hybrid: HybridNPNet | None = None
    scorer: SeedScorer | None = None

    if config.hybrid_mode in ("edit", "rerank_edit"):
        assert config.hybrid_checkpoint is not None, (
            f"--hybrid_checkpoint required for mode={config.hybrid_mode}"
        )
        hybrid = HybridNPNet(
            inference_mode=config.inference_mode,
        )
        hybrid.load_checkpoint(config.hybrid_checkpoint)
        hybrid.to(device)
        hybrid.eval()
        logger.info("HybridNPNet loaded from %s", config.hybrid_checkpoint)

    if config.hybrid_mode in ("rerank", "rerank_edit"):
        if hybrid is not None and config.scorer_checkpoint is None:
            # Use scorer from hybrid checkpoint
            scorer = hybrid.scorer
        else:
            assert config.scorer_checkpoint is not None, (
                f"--scorer_checkpoint required for mode={config.hybrid_mode}"
            )
            scorer = SeedScorer()
            scorer.load_checkpoint(config.scorer_checkpoint)
            scorer.to(device)
            scorer.eval()
        logger.info("SeedScorer ready for reranking (K=%d)", config.rerank_k)

    for category, prompts in categories:
        for seed in seeds:
            for prompt in prompts:
                dest = image_path(
                    images_dir, category, seed, prompt.prompt_id, config.image_format
                )
                if dest.exists():
                    tracker.update(1)
                    continue

                # Encode prompt
                with torch.no_grad():
                    prompt_embeds, _, _, _ = pipe.encode_prompt(
                        prompt=prompt.text, device=device
                    )

                # --- Mode: baseline ---
                if config.hybrid_mode == "baseline":
                    latent = _generate_noise_for_seed(
                        seed, config.height, config.width, device, config.generator_device
                    )

                # --- Mode: rerank ---
                elif config.hybrid_mode == "rerank":
                    assert scorer is not None
                    candidate_seeds = list(range(seed, seed + config.rerank_k))
                    best_seed = _rerank_seeds(
                        scorer, candidate_seeds, prompt_embeds,
                        config.height, config.width, device, config.generator_device,
                    )
                    latent = _generate_noise_for_seed(
                        best_seed, config.height, config.width, device, config.generator_device
                    )

                # --- Mode: edit ---
                elif config.hybrid_mode == "edit":
                    assert hybrid is not None
                    latent = _generate_noise_for_seed(
                        seed, config.height, config.width, device, config.generator_device
                    )
                    with torch.no_grad():
                        golden = hybrid(latent, prompt_embeds)
                        latent = golden.half()

                # --- Mode: rerank_edit ---
                elif config.hybrid_mode == "rerank_edit":
                    assert scorer is not None
                    assert hybrid is not None
                    candidate_seeds = list(range(seed, seed + config.rerank_k))
                    best_seed = _rerank_seeds(
                        scorer, candidate_seeds, prompt_embeds,
                        config.height, config.width, device, config.generator_device,
                    )
                    latent = _generate_noise_for_seed(
                        best_seed, config.height, config.width, device, config.generator_device
                    )
                    with torch.no_grad():
                        golden = hybrid(latent, prompt_embeds)
                        latent = golden.half()

                else:
                    raise ValueError(f"Unknown hybrid_mode: {config.hybrid_mode}")

                # Generate image
                image = pipe(
                    prompt=prompt.text,
                    latents=latent,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    height=config.height,
                    width=config.width,
                ).images[0]

                save_image_atomic(image, dest)
                tracker.update(1)

            logger.info("Seed %d [%s] done | %s", seed, category, tracker.summary_line())

    # Save run metadata
    run_meta = {
        "hybrid_mode": config.hybrid_mode,
        "inference_mode": config.inference_mode,
        "rerank_k": config.rerank_k,
        "num_seeds": len(seeds),
        "num_prompts": sum(len(p) for _, p in categories),
        "total_images": total,
    }
    meta_path = config.out_dir / "run_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(run_meta, indent=2) + "\n")

    logger.info(
        "Hybrid evaluation complete (mode=%s) | %s",
        config.hybrid_mode,
        tracker.summary_line(),
    )

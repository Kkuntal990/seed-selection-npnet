"""Core generation engine: pipeline loading, batching, OOM fallback, multi-GPU."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import torch
from PIL import Image

from seed_mining.io_utils import (
    append_metadata_jsonl,
    find_existing_images,
    image_path,
    save_image_atomic,
    save_prompts_jsonl,
    verify_completeness,
    write_done_marker,
)
from seed_mining.logging_utils import ThroughputTracker, setup_logging
from seed_mining.prompts import Prompt, build_all_prompts

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

    from seed_mining.config import SeedMiningConfig

logger = logging.getLogger("seed_mining.generator")


# ---------------------------------------------------------------------------
# Scheduler mapping
# ---------------------------------------------------------------------------

SCHEDULER_MAP: dict[str, str] = {
    "ddim": "DDIMScheduler",
    "dpmsolver++": "DPMSolverMultistepScheduler",
    "dpmsolverpp": "DPMSolverMultistepScheduler",
    "euler": "EulerDiscreteScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "pndm": "PNDMScheduler",
}


def _get_scheduler(name: str, config: dict) -> object:  # noqa: ANN401
    """Instantiate a scheduler by short name."""
    import diffusers

    cls_name = SCHEDULER_MAP.get(name.lower())
    if cls_name is None:
        raise ValueError(f"Unknown scheduler '{name}'. Choose from: {list(SCHEDULER_MAP)}")
    cls = getattr(diffusers, cls_name)
    return cls.from_config(config)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------


def _is_sd3_pipeline(pipe: DiffusionPipeline) -> bool:
    """Check if the loaded pipeline is a Stable Diffusion 3 variant."""
    cls_name = type(pipe).__name__
    return "StableDiffusion3" in cls_name


def _load_sd3_pipeline(model_id: str, dtype: torch.dtype) -> DiffusionPipeline:
    """Load a Stable Diffusion 3.x pipeline."""
    from diffusers import StableDiffusion3Pipeline

    return StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)


def _load_sd_pipeline(model_id: str, dtype: torch.dtype) -> DiffusionPipeline:
    """Load a Stable Diffusion 1.x/2.x pipeline."""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    return pipe


def load_pipeline(config: SeedMiningConfig, device: str = "cuda") -> DiffusionPipeline:
    """Load and configure a diffusion pipeline (SD 1.x/2.x/3.x)."""
    dtype = torch.float16

    # Detect pipeline type from model config on Hub
    model_id = config.model_id
    if "stable-diffusion-3" in model_id.lower() or "sd3" in model_id.lower():
        pipe = _load_sd3_pipeline(model_id, dtype)
    else:
        pipe = _load_sd_pipeline(model_id, dtype)

    # Set scheduler (SD3 uses FlowMatchEulerDiscreteScheduler — don't override)
    if _is_sd3_pipeline(pipe):
        logger.info("SD3 pipeline detected — keeping native FlowMatch scheduler")
    else:
        pipe.scheduler = _get_scheduler(config.scheduler, pipe.scheduler.config)

    # Move to device
    if config.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # Optional xformers (not supported on SD3 transformer models)
    if config.enable_xformers:
        if _is_sd3_pipeline(pipe):
            logger.warning("xformers not supported for SD3 pipelines, ignoring --enable_xformers")
        else:
            pipe.enable_xformers_memory_efficient_attention()

    pipe.set_progress_bar_config(disable=True)
    return pipe  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Batched generation with OOM fallback
# ---------------------------------------------------------------------------


def generate_batch(
    pipe: DiffusionPipeline,
    prompts: list[Prompt],
    seed: int,
    config: SeedMiningConfig,
    batch_size: int | None = None,
) -> list[Image.Image]:
    """Generate images for *prompts* at a given *seed*.

    If a CUDA OOM occurs, halves the batch size and retries recursively.
    """
    bs = batch_size or config.batch_size
    all_images: list[Image.Image] = []

    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        texts = [p.text for p in batch]

        generators = [
            torch.Generator(device=config.generator_device).manual_seed(seed) for _ in batch
        ]

        try:
            with torch.inference_mode():
                result = pipe(  # type: ignore[operator]
                    prompt=texts,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    height=config.height,
                    width=config.width,
                    generator=generators,
                )
            all_images.extend(result.images)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise RuntimeError(f"OOM even at batch_size=1 for seed={seed}") from None
            new_bs = max(1, bs // 2)
            logger.warning("OOM at batch_size=%d, retrying chunk with batch_size=%d", bs, new_bs)
            chunk_images = generate_batch(pipe, batch, seed, config, batch_size=new_bs)
            all_images.extend(chunk_images)

    return all_images


# ---------------------------------------------------------------------------
# Per-seed generation
# ---------------------------------------------------------------------------


def _build_metadata_record(
    prompt: Prompt, seed: int, rel_path: str, config: SeedMiningConfig
) -> dict:
    """Build a metadata dict for one generated image."""
    rec = {
        "task": prompt.category,
        "seed": seed,
        "prompt_id": prompt.prompt_id,
        "prompt": prompt.text,
        "image_path": rel_path,
        "model_id": config.model_id,
        "scheduler": config.scheduler,
        "steps": config.num_inference_steps,
        "guidance": config.guidance_scale,
        "height": config.height,
        "width": config.width,
        "torch_dtype": config.torch_dtype_str,
        "generator_device": config.generator_device,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **prompt.metadata,
    }
    return rec


def generate_for_seed(
    pipe: DiffusionPipeline | None,
    seed: int,
    numeracy: list[Prompt],
    spatial: list[Prompt],
    config: SeedMiningConfig,
    existing_numeracy: set[tuple[int, int]],
    existing_spatial: set[tuple[int, int]],
    tracker: ThroughputTracker,
) -> None:
    """Generate all images for one seed, skipping already-completed prompt_ids."""
    for category, prompts, existing in [
        ("numeracy", numeracy, existing_numeracy),
        ("spatial", spatial, existing_spatial),
    ]:
        todo = [p for p in prompts if (seed, p.prompt_id) not in existing]
        already = len(prompts) - len(todo)

        if not todo:
            logger.info("Seed %d [%s]: all %d images exist, skipping", seed, category, len(prompts))
            tracker.update(len(prompts))
            continue

        logger.info(
            "Seed %d [%s]: generating %d images (%d already exist)",
            seed, category, len(todo), already,
        )
        tracker.update(already)

        if config.dry_run:
            tracker.update(len(todo))
            continue

        assert pipe is not None
        images = generate_batch(pipe, todo, seed, config)

        # Save images + metadata
        metadata_records: list[dict] = []
        for prompt, img in zip(todo, images, strict=True):
            dest = image_path(config.out_dir, category, seed, prompt.prompt_id, config.image_format)
            save_image_atomic(img, dest)
            rel = str(dest.relative_to(config.out_dir))
            metadata_records.append(_build_metadata_record(prompt, seed, rel, config))
            logger.debug("Saved %s", dest)

        md_path = config.out_dir / "metadata" / f"{category}_images.jsonl"
        append_metadata_jsonl(md_path, metadata_records)
        tracker.update(len(todo))

    logger.info("Seed %d done | %s", seed, tracker.summary_line())


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_generation(config: SeedMiningConfig) -> None:
    """Run the full generation pipeline (single or multi-GPU)."""
    from accelerate import PartialState

    state = PartialState()
    rank = state.local_process_index
    world_size = state.num_processes

    log_dir = config.out_dir / "logs"
    setup_logging(log_dir, rank=rank)

    logger.info("Building prompts (split=%s) ...", config.split)
    numeracy, spatial = build_all_prompts(
        config.prompt_dataset_dir,
        config.split,
        num_objects=config.num_objects,
        num_settings=config.num_settings,
        append_background_to_spatial=config.append_background_to_spatial,
    )

    all_seeds = config.seeds
    my_seeds = [s for i, s in enumerate(all_seeds) if i % world_size == rank]

    total_per_seed = len(numeracy) + len(spatial)
    total_mine = len(my_seeds) * total_per_seed

    logger.info(
        "Rank %d/%d: %d seeds assigned | %d numeracy + %d spatial = %d prompts/seed | %d total",
        rank, world_size, len(my_seeds), len(numeracy), len(spatial), total_per_seed, total_mine,
    )

    if config.dry_run:
        if rank == 0:
            grand_total = len(all_seeds) * total_per_seed
            logger.info("=== DRY RUN SUMMARY ===")
            logger.info("Numeracy prompts per seed: %d", len(numeracy))
            logger.info("Spatial prompts per seed:  %d", len(spatial))
            logger.info("Total prompts per seed:    %d", total_per_seed)
            logger.info("Seeds:                     %d", len(all_seeds))
            logger.info("Grand total images:        %d", grand_total)
        return

    # Save config + prompts (rank 0 only)
    if rank == 0:
        config.save_resolved(config.out_dir / "run_config.json")
        save_prompts_jsonl(config.out_dir, numeracy, "numeracy")
        save_prompts_jsonl(config.out_dir, spatial, "spatial")

    state.wait_for_everyone()

    # Resume: find existing images
    existing_numeracy = find_existing_images(config.out_dir, "numeracy")
    existing_spatial = find_existing_images(config.out_dir, "spatial")
    logger.info(
        "Resume scan: %d numeracy + %d spatial images already exist",
        len(existing_numeracy), len(existing_spatial),
    )

    tracker = ThroughputTracker(total=total_mine)

    # Load pipeline
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(config, device=device)
    logger.info("Pipeline loaded on %s", device)

    for seed in my_seeds:
        generate_for_seed(
            pipe, seed, numeracy, spatial, config,
            existing_numeracy, existing_spatial, tracker,
        )

    state.wait_for_everyone()

    if rank == 0:
        summary = verify_completeness(
            config.out_dir, all_seeds, numeracy, spatial, config.image_format
        )
        logger.info("Verification: %s", summary)
        if summary["missing_numeracy"] == 0 and summary["missing_spatial"] == 0:
            write_done_marker(config.out_dir)
            logger.info("All images generated. _DONE.txt written.")
        else:
            logger.warning(
                "Generation incomplete: %d missing. Re-run to fill gaps.",
                summary["missing_numeracy"] + summary["missing_spatial"],
            )

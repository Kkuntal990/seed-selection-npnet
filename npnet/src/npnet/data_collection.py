"""Collect (source_noise, target_noise) pairs via DDIM inversion for NPNet training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline
from seed_mining.io_utils import append_metadata_jsonl
from seed_mining.logging_utils import ThroughputTracker, setup_logging
from seed_mining.prompts import Prompt, build_all_prompts

if TYPE_CHECKING:
    from npnet.config import DataCollectionConfig

logger = logging.getLogger("npnet.data_collection")


def noise_pair_path(out_dir: Path, category: str, seed: int, prompt_id: int) -> Path:
    """Canonical path for a noise pair .npz file."""
    return out_dir / category / f"seed={seed:03d}" / f"prompt={prompt_id:04d}.npz"


def save_noise_pair(
    path: Path,
    source_noise: np.ndarray,
    target_noise: np.ndarray,
    prompt_id: int,
) -> None:
    """Save a (source, target) noise pair atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, source_noise=source_noise, target_noise=target_noise, prompt_id=prompt_id)
    tmp.rename(path)


def _ddim_invert(
    pipe: Any,  # StableDiffusionXLPipeline (diffusers stubs incomplete)
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    """Run DDIM inversion: clean latent → noisy latent.

    Uses the inverse scheduler to step from z_0 back to z_T.
    """
    device = latents.device
    dtype = latents.dtype

    inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_scheduler.set_timesteps(num_steps, device=device)

    add_text_embeds = pooled_prompt_embeds
    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    add_time_ids = pipe._get_add_time_ids(
        (1024, 1024),
        (0, 0),
        (1024, 1024),
        dtype=dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    ).to(device)

    do_cfg = guidance_scale > 1.0
    if do_cfg:
        pe = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        ate = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        ati = torch.cat([add_time_ids, add_time_ids], dim=0)
    else:
        pe, ate, ati = prompt_embeds, add_text_embeds, add_time_ids

    inv_latents = latents.clone()
    for t in inv_scheduler.timesteps:
        latent_input = torch.cat([inv_latents] * 2) if do_cfg else inv_latents
        latent_input = inv_scheduler.scale_model_input(latent_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=pe,
                added_cond_kwargs={"text_embeds": ate, "time_ids": ati},
                return_dict=False,
            )[0]

        if do_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)

        inv_scheduler._step_index = None  # noqa: SLF001
        inv_latents = inv_scheduler.step(noise_pred, t, inv_latents, return_dict=False)[0]

    return inv_latents


def collect_noise_pair(
    pipe: Any,  # StableDiffusionXLPipeline (diffusers stubs incomplete)
    prompt_text: str,
    seed: int,
    num_inference_steps: int = 50,
    num_inversion_steps: int = 10,
    guidance_scale: float = 5.5,
    inversion_guidance_scale: float = 1.0,
    height: int = 1024,
    width: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a (source_noise, target_noise) pair for one prompt+seed.

    1. Sample random noise z_T with seed
    2. Generate image with DDIM scheduler
    3. Encode image to latent z_0 via VAE
    4. DDIM-invert z_0 back to z_T_inv
    5. Return (z_T, z_T_inv) as numpy arrays
    """
    device = pipe.device
    dtype = torch.float16

    # Sample source noise
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, 4, height // 8, width // 8)
    source_noise = torch.randn(shape, generator=generator, dtype=dtype).to(device)

    # Encode prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(prompt=prompt_text, device=device)

    # Forward pass: generate image
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    image = pipe(
        prompt=prompt_text,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        latents=source_noise,
    ).images[0]

    # Encode image to latent space
    image_tensor = pipe.image_processor.preprocess(image).to(device, dtype=dtype)
    with torch.no_grad():
        z_0 = pipe.vae.encode(image_tensor).latent_dist.sample()
        z_0 = z_0 * pipe.vae.config.scaling_factor

    # DDIM inversion: z_0 → z_T_inv
    target_noise = _ddim_invert(
        pipe,
        z_0,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        num_steps=num_inversion_steps,
        guidance_scale=inversion_guidance_scale,
    )

    return source_noise.cpu().numpy(), target_noise.cpu().numpy()


def run_data_collection(config: DataCollectionConfig) -> None:
    """Collect noise pairs for all prompts × seeds."""
    log_dir = config.out_dir / "logs"
    setup_logging(log_dir, rank=0)

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
        logger.info("DRY RUN: would collect %d noise pairs", total)
        return

    # Load SDXL pipeline
    logger.info("Loading pipeline: %s ...", config.model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    if config.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    # Save prompt manifest
    manifest_path = config.out_dir / "prompt_manifest.jsonl"
    manifest_records: list[dict[str, Any]] = []
    for category, prompts in categories:
        for p in prompts:
            manifest_records.append(
                {
                    "category": category,
                    "prompt_id": p.prompt_id,
                    "text": p.text,
                    "metadata": p.metadata,
                }
            )
    append_metadata_jsonl(manifest_path, manifest_records)

    for category, prompts in categories:
        for seed in seeds:
            for prompt in prompts:
                npz_path = noise_pair_path(config.out_dir, category, seed, prompt.prompt_id)
                if npz_path.exists():
                    tracker.update(1)
                    continue

                source, target = collect_noise_pair(
                    pipe,
                    prompt.text,
                    seed,
                    num_inference_steps=config.num_inference_steps,
                    num_inversion_steps=config.num_inversion_steps,
                    guidance_scale=config.guidance_scale,
                    inversion_guidance_scale=config.inversion_guidance_scale,
                    height=config.height,
                    width=config.width,
                )
                save_noise_pair(npz_path, source, target, prompt.prompt_id)
                tracker.update(1)

            logger.info("Seed %d [%s] done | %s", seed, category, tracker.summary_line())

    logger.info("Data collection complete | %s", tracker.summary_line())

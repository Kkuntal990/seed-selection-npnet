"""Generate images using golden noise from a trained NPNet."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from seed_mining.io_utils import image_path, save_image_atomic
from seed_mining.logging_utils import ThroughputTracker, setup_logging
from seed_mining.prompts import Prompt, build_all_prompts

from npnet.models.npnet import NPNet

if TYPE_CHECKING:
    from npnet.config import InferenceConfig

logger = logging.getLogger("npnet.inference")


def run_golden_generation(config: InferenceConfig) -> None:
    """Generate images using golden noise from trained NPNet."""
    log_dir = config.out_dir / "logs"
    setup_logging(log_dir, rank=0)

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
        logger.info("DRY RUN: would generate %d golden noise images", total)
        return

    # Load SDXL pipeline
    logger.info("Loading pipeline: %s ...", config.model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
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
                    config.out_dir, category, seed, prompt.prompt_id, config.image_format
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

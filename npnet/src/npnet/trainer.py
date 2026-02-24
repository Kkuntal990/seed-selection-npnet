"""NPNet training loop: MSE loss between predicted golden noise and DDIM-inverted target."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from seed_mining.logging_utils import setup_logging
from torch.utils.data import DataLoader, random_split

from npnet.dataset import NoiseDataset
from npnet.models.npnet import NPNet

if TYPE_CHECKING:
    from npnet.config import TrainingConfig

logger = logging.getLogger("npnet.trainer")


def build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    """Build train and validation data loaders."""
    dataset = NoiseDataset(config.noise_pairs_dir, config.prompt_manifest_path)

    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def run_training(config: TrainingConfig) -> None:
    """Train NPNet on collected noise pairs."""
    setup_logging(config.checkpoint_dir / "logs", rank=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    logger.info("Loading dataset from %s ...", config.noise_pairs_dir)
    train_loader, val_loader = build_dataloaders(config)
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    # SDXL pipeline (frozen, only for encode_prompt)
    logger.info("Loading SDXL pipeline for prompt encoding: %s ...", config.model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    if config.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # NPNet
    npnet = NPNet(
        latent_channels=config.latent_channels,
        latent_resolution=config.latent_resolution,
        text_embed_dim=config.text_embed_dim,
        text_seq_len=config.text_seq_len,
    ).to(device)

    if config.pretrained_path is not None:
        npnet.load_checkpoint(config.pretrained_path)

    # Optimizer
    optimizer = torch.optim.AdamW(npnet.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        # --- Train ---
        npnet.train()
        train_loss_sum = 0.0
        train_steps = 0

        for step, (source_noise, target_noise, prompts) in enumerate(train_loader, 1):
            source_noise = source_noise.to(device)
            target_noise = target_noise.to(device)

            # Encode prompts with frozen SDXL text encoders
            with torch.no_grad():
                prompt_embeds, _, _, _ = pipe.encode_prompt(
                    prompt=list(prompts),
                    device=device,
                )

            golden_noise = npnet(source_noise, prompt_embeds)
            loss = F.mse_loss(golden_noise, target_noise.float())

            # Gradient accumulation
            loss = loss / config.grad_accumulation_steps
            loss.backward()

            if step % config.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss_sum += loss.item() * config.grad_accumulation_steps
            train_steps += 1

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # --- Validate ---
        npnet.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for source_noise, target_noise, prompts in val_loader:
                source_noise = source_noise.to(device)
                target_noise = target_noise.to(device)

                prompt_embeds, _, _, _ = pipe.encode_prompt(
                    prompt=list(prompts),
                    device=device,
                )

                golden_noise = npnet(source_noise, prompt_embeds)
                loss = F.mse_loss(golden_noise, target_noise.float())

                val_loss_sum += loss.item() * len(source_noise)
                val_count += len(source_noise)

        avg_val_loss = val_loss_sum / max(val_count, 1)

        logger.info(
            "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f",
            epoch,
            config.epochs,
            avg_train_loss,
            avg_val_loss,
        )

        # Save checkpoint
        if epoch % config.save_every_n_epochs == 0:
            path = config.checkpoint_dir / f"npnet_epoch_{epoch:03d}.pth"
            npnet.save_checkpoint(path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            npnet.save_checkpoint(config.checkpoint_dir / "npnet_best.pth")
            logger.info("New best val_loss=%.6f", avg_val_loss)

    # Save final
    npnet.save_checkpoint(config.checkpoint_dir / "npnet_final.pth")
    logger.info("Training complete. Best val_loss=%.6f", best_val_loss)

"""NPNet training loop with residual + directional loss."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline
from seed_mining.logging_utils import setup_logging
from torch.utils.data import DataLoader, random_split

from npnet.dataset import NearestGoodDataset, NoiseDataset
from npnet.models.npnet import NPNet

if TYPE_CHECKING:
    from npnet.config import TrainingConfig

logger = logging.getLogger("npnet.trainer")


def residual_loss(
    predicted: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute loss on the transport delta, not full reconstruction.

    Combines:
    - Cosine similarity on flattened deltas (directional alignment, scale-invariant)
    - L1 loss on deltas (sparse gradients, better than MSE for weak signals)
    - MSE loss on full output (reconstruction fidelity)

    This gives meaningful gradients even when source/target are both ~N(0,1)
    and nearly equidistant in high dimensions.
    """
    delta_pred = predicted - source.float()
    delta_true = target.float() - source.float()

    # Directional loss: are we moving in the right direction?
    cos_loss = 1.0 - F.cosine_similarity(
        delta_pred.flatten(1), delta_true.flatten(1)
    ).mean()

    # Magnitude loss: are we moving the right amount?
    l1_loss = F.l1_loss(delta_pred, delta_true)

    # Reconstruction loss: does the output match the target?
    mse_loss = F.mse_loss(predicted, target.float())

    return cos_loss + l1_loss + 0.1 * mse_loss


def build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    """Build train and validation data loaders."""
    if config.dataset_type == "nearest_good":
        assert config.nearest_good_dir is not None
        train_ds = NearestGoodDataset(config.nearest_good_dir / "train")
        val_ds = NearestGoodDataset(config.nearest_good_dir / "val")
        logger.info(
            "Loaded nearest_good dataset: train=%d, val=%d", len(train_ds), len(val_ds)
        )
    else:
        assert config.noise_pairs_dir is not None
        assert config.prompt_manifest_path is not None
        dataset = NoiseDataset(config.noise_pairs_dir, config.prompt_manifest_path)
        val_size = int(len(dataset) * config.val_split)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(config.seed)
        train_ds, val_ds = random_split(  # type: ignore[assignment]
            dataset, [train_size, val_size], generator=generator
        )

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
    """Train NPNet on collected noise pairs.

    Supports single-GPU and multi-GPU via ``accelerate launch``.
    """
    # Accelerator handles device placement, DDP, mixed precision
    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.grad_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    is_main = accelerator.is_main_process
    device = accelerator.device

    if is_main:
        setup_logging(config.checkpoint_dir / "logs", rank=0)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    # Data
    logger.info("Loading dataset (type=%s) ...", config.dataset_type)
    train_loader, val_loader = build_dataloaders(config)
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    # SDXL pipeline (frozen, only for encode_prompt — loaded on each GPU)
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
    )

    if config.pretrained_path is not None:
        npnet.load_checkpoint(config.pretrained_path)

    # Optimizer + cosine LR scheduler with warmup
    optimizer = torch.optim.AdamW(npnet.parameters(), lr=config.lr)

    total_train_steps_per_epoch = len(train_loader)
    total_steps = total_train_steps_per_epoch * config.epochs
    warmup_steps = min(total_steps // 20, 100)  # 5% warmup, max 100 steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare with accelerator (handles DDP wrapping, device placement, data sharding)
    npnet, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        npnet, optimizer, train_loader, val_loader, scheduler
    )

    best_val_loss = float("inf")
    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_train_steps = len(train_loader)
    log_every = max(1, total_train_steps // 10)
    global_step = 0

    logger.info(
        "Training on %d GPU(s), batch_size=%d, effective_batch=%d, "
        "loss=residual(cosine+L1+MSE), warmup=%d steps",
        accelerator.num_processes,
        config.batch_size,
        config.batch_size * accelerator.num_processes * config.grad_accumulation_steps,
        warmup_steps,
    )

    for epoch in range(1, config.epochs + 1):
        # --- Train ---
        npnet.train()
        train_loss_sum = 0.0
        train_steps = 0

        for step, (source_noise, target_noise, prompts) in enumerate(train_loader, 1):
            with accelerator.accumulate(npnet):
                # Encode prompts with frozen SDXL text encoders
                with torch.no_grad():
                    prompt_embeds, _, _, _ = pipe.encode_prompt(
                        prompt=list(prompts),
                        device=device,
                    )

                golden_noise = npnet(source_noise, prompt_embeds)
                loss = residual_loss(golden_noise, source_noise, target_noise)

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            train_loss_sum += loss.item()
            train_steps += 1

            if is_main and (step % log_every == 0 or step == 1):
                avg_so_far = train_loss_sum / train_steps
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    "Epoch %d/%d [step %d/%d] loss=%.6f lr=%.2e",
                    epoch, config.epochs, step, total_train_steps, avg_so_far, current_lr,
                )

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # --- Validate ---
        npnet.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for source_noise, target_noise, prompts in val_loader:
                prompt_embeds, _, _, _ = pipe.encode_prompt(
                    prompt=list(prompts),
                    device=device,
                )

                golden_noise = npnet(source_noise, prompt_embeds)
                loss = residual_loss(golden_noise, source_noise, target_noise)

                val_loss_sum += loss.item() * len(source_noise)
                val_count += len(source_noise)

        # Aggregate val loss across GPUs
        val_loss_tensor = torch.tensor([val_loss_sum, val_count], device=device)
        val_loss_tensor = accelerator.reduce(val_loss_tensor, reduction="sum")
        avg_val_loss = val_loss_tensor[0].item() / max(val_loss_tensor[1].item(), 1)

        if is_main:
            logger.info(
                "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f",
                epoch,
                config.epochs,
                avg_train_loss,
                avg_val_loss,
            )

            # Save checkpoint (unwrap DDP model)
            unwrapped = accelerator.unwrap_model(npnet)
            if epoch % config.save_every_n_epochs == 0:
                path = config.checkpoint_dir / f"npnet_epoch_{epoch:03d}.pth"
                unwrapped.save_checkpoint(path)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                unwrapped.save_checkpoint(config.checkpoint_dir / "npnet_best.pth")
                logger.info("New best val_loss=%.6f", avg_val_loss)

    # Save final
    if is_main:
        unwrapped = accelerator.unwrap_model(npnet)
        unwrapped.save_checkpoint(config.checkpoint_dir / "npnet_final.pth")
        logger.info("Training complete. Best val_loss=%.6f", best_val_loss)

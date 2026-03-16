"""Hybrid training: Phase A (scorer) + Phase B (editor).

Phase A: Train SeedScorer on VLM ranking data with pairwise or pointwise loss.
Phase B: Train ResidualEditor with frozen NPNet + scorer, combining residual,
         scorer, attention, and regularization losses.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import StableDiffusionXLPipeline
from seed_mining.logging_utils import setup_logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from npnet.datasets.seed_score_dataset import (
    PairwiseSeedScoreDataset,
    PointwiseSeedScoreDataset,
    PrecomputedEmbedPointwiseDataset,
    PrecomputedEmbedScorerDataset,
    get_scorer_prompt_ids,
)
from npnet.models.hybrid_npnet import HybridNPNet
from npnet.models.seed_scorer import SeedScorer
from npnet.trainer import PrecomputedEmbedDataset, build_datasets, residual_loss

if TYPE_CHECKING:
    from npnet.config import HybridTrainingConfig, ScorerConfig

logger = logging.getLogger("npnet.hybrid_trainer")


# ---------------------------------------------------------------------------
# Shared: prompt embedding pre-computation for scorer datasets
# ---------------------------------------------------------------------------


def precompute_scorer_embeddings(
    pipe: StableDiffusionXLPipeline,
    dataset: PairwiseSeedScoreDataset | PointwiseSeedScoreDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Pre-compute SDXL prompt embeddings for all unique prompts in a scorer dataset."""
    unique_prompts: set[str] = set()
    if isinstance(dataset, PairwiseSeedScoreDataset):
        for _, _, text in dataset.pairs:
            unique_prompts.add(text)
    else:
        for _, text, _ in dataset.samples:
            unique_prompts.add(text)

    logger.info("Pre-computing embeddings for %d unique scorer prompts ...", len(unique_prompts))
    prompt_list = sorted(unique_prompts)
    cache: dict[str, torch.Tensor] = {}

    num_batches = (len(prompt_list) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, len(prompt_list), batch_size),
            total=num_batches,
            desc="Encoding scorer prompts",
        ):
            batch = prompt_list[i : i + batch_size]
            embeds, _, _, _ = pipe.encode_prompt(prompt=batch, device=device)
            for j, p in enumerate(batch):
                cache[p] = embeds[j].cpu()

    logger.info("Scorer prompt embedding cache ready (%d entries)", len(cache))
    return cache


# ---------------------------------------------------------------------------
# Phase A: Scorer training
# ---------------------------------------------------------------------------


def run_scorer_training(config: ScorerConfig) -> None:
    """Train SeedScorer on VLM ranking data.

    Supports pairwise (MarginRankingLoss) and pointwise (BCEWithLogitsLoss).
    """
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    is_main = accelerator.is_main_process
    device = accelerator.device

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(config.checkpoint_dir / "logs", rank=0)

    # Split prompt IDs
    train_pids, val_pids, _ = get_scorer_prompt_ids(
        config.ranking_dir,
        config.categories,
        seed=config.seed,
    )
    logger.info("Prompt splits: train=%d, val=%d", len(train_pids), len(val_pids))

    # Build datasets
    if config.mode == "pairwise":
        train_ds = PairwiseSeedScoreDataset(
            config.ranking_dir,
            config.categories,
            prompt_ids=train_pids,
            height=config.height,
            width=config.width,
            max_pairs_per_prompt=config.max_pairs_per_prompt,
        )
        val_ds = PairwiseSeedScoreDataset(
            config.ranking_dir,
            config.categories,
            prompt_ids=val_pids,
            height=config.height,
            width=config.width,
            max_pairs_per_prompt=config.max_pairs_per_prompt,
        )
    else:
        train_ds_raw = PointwiseSeedScoreDataset(
            config.ranking_dir,
            config.categories,
            prompt_ids=train_pids,
            height=config.height,
            width=config.width,
        )
        val_ds_raw = PointwiseSeedScoreDataset(
            config.ranking_dir,
            config.categories,
            prompt_ids=val_pids,
            height=config.height,
            width=config.width,
        )

    # Load SDXL for prompt encoding, then discard
    logger.info("Loading SDXL for prompt encoding: %s ...", config.model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to(device)

    if config.mode == "pairwise":
        train_cache = precompute_scorer_embeddings(pipe, train_ds, device)
        val_cache = precompute_scorer_embeddings(pipe, val_ds, device)
        train_embed_ds = PrecomputedEmbedScorerDataset(train_ds, train_cache)
        val_embed_ds = PrecomputedEmbedScorerDataset(val_ds, val_cache)
    else:
        train_cache = precompute_scorer_embeddings(pipe, train_ds_raw, device)
        val_cache = precompute_scorer_embeddings(pipe, val_ds_raw, device)
        train_embed_ds = PrecomputedEmbedPointwiseDataset(train_ds_raw, train_cache)  # type: ignore[assignment]
        val_embed_ds = PrecomputedEmbedPointwiseDataset(val_ds_raw, val_cache)  # type: ignore[assignment]

    del pipe
    torch.cuda.empty_cache()
    logger.info("SDXL unloaded, VRAM freed for scorer training")

    train_loader = DataLoader(
        train_embed_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_embed_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    scorer = SeedScorer(
        latent_channels=4,
        text_embed_dim=config.text_embed_dim,
        text_seq_len=config.text_seq_len,
        hidden_dim=config.scorer_hidden_dim,
    )

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=config.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = min(total_steps // 20, 100)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scorer, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        scorer, optimizer, train_loader, val_loader, scheduler
    )

    best_val_metric = float("-inf") if config.mode == "pairwise" else float("inf")
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        scorer.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch in train_loader:
            if config.mode == "pairwise":
                good_noise, bad_noise, prompt_embeds = batch
                good_score = scorer(good_noise, prompt_embeds)
                bad_score = scorer(bad_noise, prompt_embeds)
                target = torch.ones(good_score.shape[0], device=device)
                loss = F.margin_ranking_loss(
                    good_score.squeeze(-1),
                    bad_score.squeeze(-1),
                    target,
                    margin=config.margin,
                )
            else:
                noise, prompt_embeds, labels = batch
                score = scorer(noise, prompt_embeds)
                loss = F.binary_cross_entropy_with_logits(score, labels)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # Validate
        scorer.eval()
        val_metric_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                if config.mode == "pairwise":
                    good_noise, bad_noise, prompt_embeds = batch
                    good_score = scorer(good_noise, prompt_embeds)
                    bad_score = scorer(bad_noise, prompt_embeds)
                    # Pairwise ranking accuracy
                    correct = (good_score > bad_score).float().sum().item()
                    val_metric_sum += correct
                    val_count += good_score.shape[0]
                else:
                    noise, prompt_embeds, labels = batch
                    score = scorer(noise, prompt_embeds)
                    loss = F.binary_cross_entropy_with_logits(score, labels)
                    val_metric_sum += loss.item() * noise.shape[0]
                    val_count += noise.shape[0]

        # Aggregate across GPUs
        val_tensor = torch.tensor([val_metric_sum, val_count], device=device)
        val_tensor = accelerator.reduce(val_tensor, reduction="sum")
        avg_val_metric = val_tensor[0].item() / max(val_tensor[1].item(), 1)

        if is_main:
            if config.mode == "pairwise":
                logger.info(
                    "Epoch %d/%d: train_loss=%.6f, val_ranking_acc=%.4f",
                    epoch,
                    config.epochs,
                    avg_train_loss,
                    avg_val_metric,
                )
                improved = avg_val_metric > best_val_metric
            else:
                logger.info(
                    "Epoch %d/%d: train_loss=%.6f, val_bce_loss=%.6f",
                    epoch,
                    config.epochs,
                    avg_train_loss,
                    avg_val_metric,
                )
                improved = avg_val_metric < best_val_metric

            unwrapped = accelerator.unwrap_model(scorer)
            if epoch % config.save_every_n_epochs == 0:
                unwrapped.save_checkpoint(
                    config.checkpoint_dir / f"scorer_epoch_{epoch:03d}.pth"
                )

            if improved:
                best_val_metric = avg_val_metric
                epochs_without_improvement = 0
                unwrapped.save_checkpoint(config.checkpoint_dir / "scorer_best.pth")
                logger.info("New best: %.6f", avg_val_metric)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No improvement for %d epoch(s) (patience=%d)",
                    epochs_without_improvement,
                    config.early_stopping_patience,
                )

        # Early stopping
        should_stop = torch.tensor(
            [1 if epochs_without_improvement >= config.early_stopping_patience else 0],
            device=device,
        )
        should_stop = accelerator.reduce(should_stop, reduction="max")
        if should_stop.item() >= 1:
            if is_main:
                logger.info("Early stopping at epoch %d", epoch)
            break

    if is_main:
        unwrapped = accelerator.unwrap_model(scorer)
        unwrapped.save_checkpoint(config.checkpoint_dir / "scorer_final.pth")
        logger.info("Scorer training complete. Best metric=%.6f", best_val_metric)


# ---------------------------------------------------------------------------
# Phase B: Hybrid editor training
# ---------------------------------------------------------------------------


def run_hybrid_training(config: HybridTrainingConfig) -> None:
    """Train ResidualEditor with frozen NPNet + scorer.

    Loss = lambda_edit * residual_loss + lambda_score * (-score)
         + lambda_norm * ||delta||^2 + lambda_attn * attn_loss
    """
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.grad_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    is_main = accelerator.is_main_process
    device = accelerator.device

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(config.checkpoint_dir / "logs", rank=0)

    # Build datasets (reuse existing trainer infrastructure)
    logger.info("Loading dataset (type=%s) ...", config.dataset_type)
    train_ds, val_ds = build_datasets(config)  # type: ignore[arg-type]

    # Load SDXL for prompt encoding
    logger.info("Loading SDXL for prompt encoding: %s ...", config.model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to(device)

    # Pre-compute embeddings
    from npnet.trainer import precompute_prompt_embeddings

    embed_cache = precompute_prompt_embeddings(pipe, config.dataset_dir, device)  # type: ignore[arg-type]
    train_ds = PrecomputedEmbedDataset(train_ds, embed_cache)
    val_ds = PrecomputedEmbedDataset(val_ds, embed_cache)

    # Keep frozen UNet for attention loss if needed
    frozen_unet = None
    pooled_cache: dict[str, torch.Tensor] | None = None
    if config.lambda_attn > 0 and config.attn_loss_type == "denoising_consistency":
        logger.info("Keeping frozen UNet for denoising consistency loss")
        frozen_unet = pipe.unet
        frozen_unet.requires_grad_(False)
        frozen_unet.eval()
        # Pre-compute pooled embeddings
        pooled_cache = {}
        unique_prompts = sorted(embed_cache.keys())
        with torch.no_grad():
            for i in range(0, len(unique_prompts), 64):
                batch = unique_prompts[i : i + 64]
                _, _, pooled, _ = pipe.encode_prompt(prompt=batch, device=device)
                for j, p in enumerate(batch):
                    pooled_cache[p] = pooled[j].cpu()
        # Delete everything except UNet
        pipe.vae = None  # type: ignore[assignment]
        pipe.text_encoder = None  # type: ignore[assignment]
        pipe.text_encoder_2 = None  # type: ignore[assignment]
        pipe.tokenizer = None  # type: ignore[assignment]
        pipe.tokenizer_2 = None  # type: ignore[assignment]
    else:
        del pipe

    torch.cuda.empty_cache()

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

    # Build HybridNPNet
    hybrid = HybridNPNet(
        latent_channels=config.latent_channels,
        latent_resolution=config.latent_resolution,
        text_embed_dim=config.text_embed_dim,
        text_seq_len=config.text_seq_len,
        inference_mode=config.inference_mode,
    )

    # Load pretrained components
    if config.npnet_checkpoint is not None:
        hybrid.npnet.load_checkpoint(config.npnet_checkpoint)
    if config.scorer_checkpoint is not None:
        hybrid.scorer.load_checkpoint(config.scorer_checkpoint)

    # Freeze NPNet and scorer
    if config.freeze_npnet:
        hybrid.npnet.requires_grad_(False)
        logger.info("NPNet frozen (%d params)", sum(p.numel() for p in hybrid.npnet.parameters()))
    if config.freeze_scorer:
        hybrid.scorer.requires_grad_(False)
        logger.info(
            "Scorer frozen (%d params)", sum(p.numel() for p in hybrid.scorer.parameters())
        )

    # Only optimize editor parameters
    trainable_params = [p for p in hybrid.parameters() if p.requires_grad]
    logger.info(
        "Trainable parameters: %d (editor: %d)",
        sum(p.numel() for p in trainable_params),
        sum(p.numel() for p in hybrid.editor.parameters()),
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = min(total_steps // 20, 100)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare with accelerator
    hybrid, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        hybrid, optimizer, train_loader, val_loader, scheduler
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    total_train_steps = len(train_loader)
    log_every = max(1, total_train_steps // 10)

    logger.info(
        "Hybrid training: lambda_edit=%.3f, lambda_score=%.3f, "
        "lambda_attn=%.3f, lambda_norm=%.3f, attn_loss=%s",
        config.lambda_edit,
        config.lambda_score,
        config.lambda_attn,
        config.lambda_norm,
        config.attn_loss_type,
    )

    for epoch in range(1, config.epochs + 1):
        hybrid.train()
        # Keep frozen components in eval mode
        unwrapped = accelerator.unwrap_model(hybrid)
        if config.freeze_npnet:
            unwrapped.npnet.eval()
        if config.freeze_scorer:
            unwrapped.scorer.eval()

        train_loss_sum = 0.0
        train_steps = 0

        for step, (source_noise, target, prompt_embeds) in enumerate(train_loader, 1):
            with accelerator.accumulate(hybrid):
                prompt_embeds = prompt_embeds.to(device)

                components = hybrid(source_noise, prompt_embeds, return_components=True)
                assert isinstance(components, dict)
                golden = components["golden_noise"]
                delta = components["editor_delta"]
                score = components["score"]

                # Loss components
                loss = torch.tensor(0.0, device=device)

                if config.lambda_edit > 0:
                    l_edit = residual_loss(golden, source_noise, target)
                    loss = loss + config.lambda_edit * l_edit

                if config.lambda_score > 0:
                    # Maximize score (minimize negative mean score)
                    l_score = -score.mean()
                    loss = loss + config.lambda_score * l_score

                if config.lambda_norm > 0:
                    l_norm = (delta ** 2).mean()
                    loss = loss + config.lambda_norm * l_norm

                if (
                    config.lambda_attn > 0
                    and frozen_unet is not None
                    and config.attn_loss_type == "denoising_consistency"
                ):
                    from npnet.utils.attention_metrics import (
                        denoising_consistency_loss,
                        sample_random_timestep,
                    )

                    timestep = sample_random_timestep(golden.shape[0], device=device)
                    # We need pooled embeddings for SDXL UNet
                    # Use zeros as fallback if pooled cache not available
                    pooled = torch.zeros(golden.shape[0], 1280, device=device)
                    l_attn = denoising_consistency_loss(
                        frozen_unet,
                        golden.float(),
                        target.float(),
                        prompt_embeds.float(),
                        pooled,
                        timestep,
                    )
                    loss = loss + config.lambda_attn * l_attn

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_sum += loss.item()
            train_steps += 1

            if is_main and (step % log_every == 0 or step == 1):
                avg_so_far = train_loss_sum / train_steps
                logger.info(
                    "Epoch %d/%d [step %d/%d] loss=%.6f lr=%.2e",
                    epoch,
                    config.epochs,
                    step,
                    total_train_steps,
                    avg_so_far,
                    scheduler.get_last_lr()[0],
                )

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # Validate
        hybrid.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for source_noise, target, prompt_embeds in val_loader:
                prompt_embeds = prompt_embeds.to(device)
                golden = hybrid(source_noise, prompt_embeds)
                loss = residual_loss(golden, source_noise, target)
                val_loss_sum += loss.item() * len(source_noise)
                val_count += len(source_noise)

        val_tensor = torch.tensor([val_loss_sum, val_count], device=device)
        val_tensor = accelerator.reduce(val_tensor, reduction="sum")
        avg_val_loss = val_tensor[0].item() / max(val_tensor[1].item(), 1)

        if is_main:
            logger.info(
                "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f",
                epoch,
                config.epochs,
                avg_train_loss,
                avg_val_loss,
            )

            unwrapped = accelerator.unwrap_model(hybrid)
            if epoch % config.save_every_n_epochs == 0:
                unwrapped.save_checkpoint(
                    config.checkpoint_dir / f"hybrid_epoch_{epoch:03d}.pth"
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                unwrapped.save_checkpoint(config.checkpoint_dir / "hybrid_best.pth")
                logger.info("New best val_loss=%.6f", avg_val_loss)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No improvement for %d epoch(s) (patience=%d)",
                    epochs_without_improvement,
                    config.early_stopping_patience,
                )

        should_stop = torch.tensor(
            [1 if epochs_without_improvement >= config.early_stopping_patience else 0],
            device=device,
        )
        should_stop = accelerator.reduce(should_stop, reduction="max")
        if should_stop.item() >= 1:
            if is_main:
                logger.info("Early stopping at epoch %d", epoch)
            break

    if is_main:
        unwrapped = accelerator.unwrap_model(hybrid)
        unwrapped.save_checkpoint(config.checkpoint_dir / "hybrid_final.pth")
        logger.info("Hybrid training complete. Best val_loss=%.6f", best_val_loss)

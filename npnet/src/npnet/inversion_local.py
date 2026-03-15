"""Build inversion-local perturbation training pairs.

For each golden (prompt, seed) pair:
1. Generate image with SDXL
2. DDIM-invert to recover golden noise z*
3. Create N perturbations: source = z* + ε, target = z*

This produces a local denoising task where NPNet learns to remove
small perturbations from golden noise latents.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from tqdm import tqdm

from npnet.data_collection import _ddim_invert

logger = logging.getLogger("npnet.inversion_local")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InversionLocalConfig:
    """Configuration for building inversion-local training pairs."""

    ranking_dir: Path
    out_dir: Path
    categories: list[str] = field(default_factory=lambda: ["numeracy", "spatial"])
    min_accuracy: float = 0.10
    max_accuracy: float = 0.90
    top_k: int = 3
    num_perturbations: int = 5
    perturbation_sigma: float = 0.05

    # SDXL generation / inversion params
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 50
    num_inversion_steps: int = 10
    guidance_scale: float = 5.5
    inversion_guidance_scale: float = 1.0
    height: int = 1024
    width: int = 1024

    # Split
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42

    # Runtime
    enable_cpu_offload: bool = True

    def __post_init__(self) -> None:
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")


# ---------------------------------------------------------------------------
# Loading ranking data (reused from delta_noise.py)
# ---------------------------------------------------------------------------


def load_seed_prompt_matrix(csv_path: Path) -> dict[int, dict[int, bool]]:
    """Load seed_prompt_matrix.csv -> {seed: {prompt_id: correct}}."""
    matrix: dict[int, dict[int, bool]] = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        assert all(col.startswith("p") for col in header[1:]), (
            f"Expected columns p0,p1,..., got: {header[1:5]}"
        )
        prompt_ids = [int(col[1:]) for col in header[1:]]
        for row in reader:
            seed = int(row[0])
            matrix[seed] = {pid: val == "1" for pid, val in zip(prompt_ids, row[1:], strict=False)}
    return matrix


def load_prompt_accuracy(json_path: Path) -> dict[int, float]:
    """Load prompt_accuracy.json -> {prompt_id: accuracy}."""
    data = json.loads(json_path.read_text())
    return {p["prompt_id"]: p["accuracy"] for p in data["prompts_ranked"]}


def load_prompt_texts(json_path: Path) -> dict[int, str]:
    """Load prompt_accuracy.json -> {prompt_id: prompt_text}."""
    data = json.loads(json_path.read_text())
    return {p["prompt_id"]: p["prompt_text"] for p in data["prompts_ranked"]}


def load_seed_overall_accuracy(json_path: Path) -> dict[int, float]:
    """Load ranked_seeds.json -> {seed: accuracy}."""
    data = json.loads(json_path.read_text())
    return {s["seed"]: s["accuracy"] for s in data["seeds_ranked"]}


# ---------------------------------------------------------------------------
# Filtering & selection
# ---------------------------------------------------------------------------


def filter_prompts_by_difficulty(
    prompt_accuracies: dict[int, float],
    min_accuracy: float = 0.10,
    max_accuracy: float = 0.90,
) -> list[int]:
    """Return prompt_ids with accuracy in [min_accuracy, max_accuracy]."""
    return sorted(
        pid for pid, acc in prompt_accuracies.items() if min_accuracy <= acc <= max_accuracy
    )


def get_good_seeds_for_prompt(
    matrix: dict[int, dict[int, bool]],
    prompt_id: int,
    top_k: int,
    seed_overall_acc: dict[int, float],
) -> list[int]:
    """Return top-K seeds that are correct for *prompt_id*."""
    correct_seeds = [seed for seed, pmap in matrix.items() if pmap.get(prompt_id, False)]
    if not correct_seeds:
        return []
    correct_seeds.sort(key=lambda s: (-seed_overall_acc.get(s, 0.0), s))
    return correct_seeds[:top_k]


def split_prompts_deterministic(
    prompt_ids: list[int],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split *prompt_ids* into train / val / test deterministically."""
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(prompt_ids), generator=rng).tolist()
    n_train = int(len(prompt_ids) * train_frac)
    n_val = int(len(prompt_ids) * val_frac)
    train_ids = sorted(prompt_ids[i] for i in indices[:n_train])
    val_ids = sorted(prompt_ids[i] for i in indices[n_train : n_train + n_val])
    test_ids = sorted(prompt_ids[i] for i in indices[n_train + n_val :])
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# DDIM inversion for golden noise
# ---------------------------------------------------------------------------


def invert_golden_seed(
    pipe: Any,
    prompt_text: str,
    seed: int,
    num_inference_steps: int = 50,
    num_inversion_steps: int = 10,
    guidance_scale: float = 5.5,
    inversion_guidance_scale: float = 1.0,
    height: int = 1024,
    width: int = 1024,
) -> torch.Tensor:
    """Generate image from golden seed and DDIM-invert to recover golden noise z*.

    Returns z* as a float32 tensor of shape (4, H/8, W/8).
    """
    device = pipe.device
    dtype = torch.float16

    # Sample noise with golden seed
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, 4, height // 8, width // 8)
    source_noise = torch.randn(shape, generator=generator, dtype=dtype).to(device)

    # Encode prompt (get all 4 embedding tensors for DDIM inversion)
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(prompt=prompt_text, device=device)

    # Forward pass: generate image with DDIM scheduler
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

    # DDIM inversion: z_0 → z* (golden noise)
    z_star = _ddim_invert(
        pipe,
        z_0,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        num_steps=num_inversion_steps,
        guidance_scale=inversion_guidance_scale,
    )

    return z_star.squeeze(0).float()  # (4, H/8, W/8)


# ---------------------------------------------------------------------------
# Perturbation pair building
# ---------------------------------------------------------------------------


def build_perturbation_pairs(
    z_star: torch.Tensor,
    prompt_text: str,
    seed: int,
    prompt_id: int,
    category: str,
    num_perturbations: int = 5,
    sigma: float = 0.05,
    rng: torch.Generator | None = None,
) -> list[dict[str, Any]]:
    """Create N perturbation pairs from a single golden noise z*.

    Each pair: source = z* + ε, target = z*, where ε ~ N(0, σ²I).
    Returns list of dicts ready for torch.save().
    """
    pairs: list[dict[str, Any]] = []
    for i in range(num_perturbations):
        epsilon = sigma * torch.randn_like(z_star, generator=rng)
        source = z_star + epsilon
        epsilon_norm = torch.norm(epsilon, p=2).item()

        pairs.append(
            {
                "source_noise": source,
                "target_noise": z_star.clone(),
                "prompt_text": prompt_text,
                "seed": seed,
                "prompt_id": prompt_id,
                "category": category,
                "perturbation_idx": i,
                "epsilon_norm": epsilon_norm,
                "sigma": sigma,
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_build_inversion_local_pairs(config: InversionLocalConfig) -> None:
    """Build inversion-local perturbation pairs for all categories."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    rng = torch.Generator().manual_seed(config.seed)

    # Save run config
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "config.json").write_text(
        json.dumps(
            {
                "ranking_dir": str(config.ranking_dir),
                "out_dir": str(config.out_dir),
                "categories": config.categories,
                "min_accuracy": config.min_accuracy,
                "max_accuracy": config.max_accuracy,
                "top_k": config.top_k,
                "num_perturbations": config.num_perturbations,
                "perturbation_sigma": config.perturbation_sigma,
                "model_id": config.model_id,
                "num_inference_steps": config.num_inference_steps,
                "num_inversion_steps": config.num_inversion_steps,
                "guidance_scale": config.guidance_scale,
                "inversion_guidance_scale": config.inversion_guidance_scale,
                "height": config.height,
                "width": config.width,
                "train_frac": config.train_frac,
                "val_frac": config.val_frac,
                "seed": config.seed,
            },
            indent=2,
        )
        + "\n"
    )

    # Load SDXL pipeline
    logger.info("Loading SDXL pipeline: %s ...", config.model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    # Always load full pipeline on GPU — cpu_offload breaks manual VAE
    # encoding and DDIM inversion hooks. A100 80GB has plenty of VRAM.
    pipe.to("cuda")

    total_pairs = 0

    # Clear existing manifests
    for split_name in ("train", "val", "test"):
        manifest_path = config.out_dir / split_name / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

    for category in config.categories:
        cat_dir = config.ranking_dir / category
        if not cat_dir.exists():
            logger.warning("Ranking directory %s does not exist, skipping", cat_dir)
            continue

        # 1. Load ranking data
        matrix = load_seed_prompt_matrix(cat_dir / "seed_prompt_matrix.csv")
        prompt_acc = load_prompt_accuracy(cat_dir / "prompt_accuracy.json")
        prompt_texts = load_prompt_texts(cat_dir / "prompt_accuracy.json")
        seed_overall_acc = load_seed_overall_accuracy(cat_dir / "ranked_seeds.json")

        logger.info("[%s] Loaded %d seeds, %d prompts", category, len(matrix), len(prompt_acc))

        # 2. Filter prompts by difficulty
        filtered_pids = filter_prompts_by_difficulty(
            prompt_acc, config.min_accuracy, config.max_accuracy
        )
        logger.info(
            "[%s] %d / %d prompts pass difficulty filter [%.2f, %.2f]",
            category,
            len(filtered_pids),
            len(prompt_acc),
            config.min_accuracy,
            config.max_accuracy,
        )

        if not filtered_pids:
            logger.warning("[%s] No prompts passed difficulty filter", category)
            continue

        # 3. Split prompts
        train_pids, val_pids, test_pids = split_prompts_deterministic(
            filtered_pids, config.train_frac, config.val_frac, config.seed
        )
        logger.info(
            "[%s] Split: train=%d, val=%d, test=%d",
            category,
            len(train_pids),
            len(val_pids),
            len(test_pids),
        )

        # Save filtered prompts info
        (config.out_dir / f"{category}_filtered_prompts.json").write_text(
            json.dumps(
                {
                    "category": category,
                    "total_prompts": len(prompt_acc),
                    "filtered_prompts": len(filtered_pids),
                    "train": train_pids,
                    "val": val_pids,
                    "test": test_pids,
                },
                indent=2,
            )
            + "\n"
        )

        # 4. Build pairs for each split
        for split_name, pids in [
            ("train", train_pids),
            ("val", val_pids),
            ("test", test_pids),
        ]:
            manifest_records: list[dict[str, Any]] = []

            for pid in tqdm(pids, desc=f"{category}/{split_name}", unit="prompt"):
                prompt_text = prompt_texts.get(pid, "")
                golden_seeds = get_good_seeds_for_prompt(
                    matrix, pid, config.top_k, seed_overall_acc
                )

                if not golden_seeds:
                    logger.warning("[%s] prompt %d has no correct seeds, skipping", category, pid)
                    continue

                for golden_seed in golden_seeds:
                    # Check if already done (resume support)
                    first_pt = (
                        config.out_dir
                        / split_name
                        / category
                        / f"prompt_{pid:04d}"
                        / f"seed={golden_seed:03d}_pert=0.pt"
                    )
                    if first_pt.exists():
                        # Assume all perturbations for this seed are done
                        for pi in range(config.num_perturbations):
                            rel_path = (
                                f"{category}/prompt_{pid:04d}/seed={golden_seed:03d}_pert={pi}.pt"
                            )
                            manifest_records.append(
                                {
                                    "path": rel_path,
                                    "seed": golden_seed,
                                    "prompt_id": pid,
                                    "category": category,
                                    "perturbation_idx": pi,
                                }
                            )
                        continue

                    # DDIM inversion to get golden noise z*
                    logger.info(
                        "[%s/%s] Inverting prompt=%d seed=%d ...",
                        category,
                        split_name,
                        pid,
                        golden_seed,
                    )
                    z_star = invert_golden_seed(
                        pipe,
                        prompt_text,
                        golden_seed,
                        num_inference_steps=config.num_inference_steps,
                        num_inversion_steps=config.num_inversion_steps,
                        guidance_scale=config.guidance_scale,
                        inversion_guidance_scale=config.inversion_guidance_scale,
                        height=config.height,
                        width=config.width,
                    )

                    # Generate perturbation pairs
                    pairs = build_perturbation_pairs(
                        z_star,
                        prompt_text,
                        golden_seed,
                        pid,
                        category,
                        num_perturbations=config.num_perturbations,
                        sigma=config.perturbation_sigma,
                        rng=rng,
                    )

                    for pair_data in pairs:
                        pi = pair_data["perturbation_idx"]
                        rel_path = (
                            f"{category}/prompt_{pid:04d}/seed={golden_seed:03d}_pert={pi}.pt"
                        )
                        pt_path = config.out_dir / split_name / rel_path
                        pt_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(pair_data, pt_path)

                        manifest_records.append(
                            {
                                "path": rel_path,
                                "seed": golden_seed,
                                "prompt_id": pid,
                                "category": category,
                                "perturbation_idx": pi,
                                "epsilon_norm": pair_data["epsilon_norm"],
                            }
                        )

            # Write manifest
            manifest_path = config.out_dir / split_name / "manifest.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("a") as f:
                for rec in manifest_records:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")

            total_pairs += len(manifest_records)
            logger.info("[%s/%s] Wrote %d pairs", category, split_name, len(manifest_records))

    logger.info("Done. Total pairs: %d", total_pairs)

"""Check if NPNet collapsed to predicting -z_source (cancellation mode).

Check 1: cos(delta_hat, -z_source) — if ~1.0, model cancels input
Check 2: ||z + delta_hat|| / ||z||  — if ~0.0, corrected latent is near-zero
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline

from npnet.dataset import DeltaNoiseDataset
from npnet.models.npnet import NPNet


def main() -> None:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/output/npnet_checkpoints/delta/npnet_best.pth")
    dataset_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/output/delta_noise/val")
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading NPNet from {checkpoint} ...")
    npnet = NPNet(latent_channels=4, latent_resolution=128, text_embed_dim=2048, text_seq_len=77)
    npnet.load_checkpoint(checkpoint)
    npnet.to(device)
    npnet.eval()

    # Load SDXL for prompt encoding
    print("Loading SDXL pipeline for prompt encoding ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()

    # Load dataset
    print(f"Loading dataset from {dataset_dir} ...")
    ds = DeltaNoiseDataset(dataset_dir)
    n = min(max_samples, len(ds))
    print(f"Running checks on {n} samples ...\n")

    cos_with_neg_source = []
    norm_ratios = []
    cos_with_true_delta = []
    pred_norms = []
    source_norms = []

    with torch.no_grad():
        for i in range(n):
            source_noise, true_delta, prompt_text = ds[i]
            source = source_noise.unsqueeze(0).to(device)  # (1, C, H, W)

            prompt_embeds, _, _, _ = pipe.encode_prompt(prompt=prompt_text, device=device)
            delta_hat = npnet(source, prompt_embeds)  # predicted delta

            # Flatten for cosine similarity
            delta_flat = delta_hat.float().flatten()
            source_flat = source.float().flatten()
            true_delta_flat = true_delta.flatten().to(device)

            # Check 1: cos(delta_hat, -z_source)
            cos_cancel = F.cosine_similarity(delta_flat.unsqueeze(0), -source_flat.unsqueeze(0)).item()
            cos_with_neg_source.append(cos_cancel)

            # Check 2: ||z + delta_hat|| / ||z||
            corrected = source.float() + delta_hat.float()
            ratio = corrected.flatten().norm() / source_flat.norm()
            norm_ratios.append(ratio.item())

            # Bonus: cos(delta_hat, true_delta)
            cos_true = F.cosine_similarity(delta_flat.unsqueeze(0), true_delta_flat.unsqueeze(0)).item()
            cos_with_true_delta.append(cos_true)

            # Norms
            pred_norms.append(delta_flat.norm().item())
            source_norms.append(source_flat.norm().item())

    # Report
    def stats(vals: list[float]) -> str:
        t = torch.tensor(vals)
        return f"mean={t.mean():.4f}, std={t.std():.4f}, min={t.min():.4f}, max={t.max():.4f}"

    print("=" * 60)
    print("CHECK 1: cos(delta_hat, -z_source)")
    print(f"  {stats(cos_with_neg_source)}")
    if torch.tensor(cos_with_neg_source).mean() > 0.8:
        print("  >>> COLLAPSE CONFIRMED: model is canceling the input")
    elif torch.tensor(cos_with_neg_source).mean() > 0.5:
        print("  >>> PARTIAL COLLAPSE: model trends toward cancellation")
    else:
        print("  >>> No cancellation detected")

    print()
    print("CHECK 2: ||z + delta_hat|| / ||z||")
    print(f"  {stats(norm_ratios)}")
    if torch.tensor(norm_ratios).mean() < 0.3:
        print("  >>> COLLAPSE CONFIRMED: corrected latent is near-zero")
    elif torch.tensor(norm_ratios).mean() < 0.7:
        print("  >>> PARTIAL COLLAPSE: corrected latent is significantly smaller")
    else:
        print("  >>> Norm preserved")

    print()
    print("BONUS: cos(delta_hat, true_delta)")
    print(f"  {stats(cos_with_true_delta)}")

    print()
    print(f"Predicted delta norm:  {stats(pred_norms)}")
    print(f"Source noise norm:     {stats(source_norms)}")


if __name__ == "__main__":
    main()

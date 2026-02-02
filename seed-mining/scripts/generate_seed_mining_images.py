#!/usr/bin/env python
"""Wrapper script for seed-mining image generation.

Usage (single GPU):
    python scripts/generate_seed_mining_images.py --out_dir ./output --split all

Usage (multi-GPU via accelerate):
    accelerate launch --num_processes 4 scripts/generate_seed_mining_images.py \
        --out_dir ./output --split all

Usage (dry run):
    python scripts/generate_seed_mining_images.py --out_dir /tmp/out --dry_run
"""

from seed_mining.cli import main

if __name__ == "__main__":
    main()

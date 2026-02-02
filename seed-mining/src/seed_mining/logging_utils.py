"""Logging setup and throughput tracking."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_dir: Path, rank: int = 0) -> logging.Logger:
    """Configure the ``seed_mining`` logger.

    Rank 0 gets a Rich console handler; all ranks get a file handler.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("seed_mining")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-init
    if logger.handlers:
        return logger

    if rank == 0:
        console = RichHandler(rich_tracebacks=True, show_time=True)
        console.setLevel(logging.INFO)
        logger.addHandler(console)

    fh = logging.FileHandler(log_dir / f"generation_rank{rank}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)

    return logger


class ThroughputTracker:
    """Track generation throughput and estimate remaining time."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self._start = time.monotonic()

    def update(self, n: int = 1) -> dict[str, float]:
        """Record *n* completed items and return current stats."""
        self.done += n
        elapsed = time.monotonic() - self._start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.done
        eta = remaining / rate if rate > 0 else float("inf")
        return {
            "done": self.done,
            "total": self.total,
            "rate_per_sec": round(rate, 2),
            "eta_seconds": round(eta, 1),
            "elapsed_seconds": round(elapsed, 1),
        }

    def summary_line(self) -> str:
        """Return a human-readable progress string."""
        s = self.update(0)
        pct = s["done"] / s["total"] * 100 if s["total"] else 0
        eta_min = s["eta_seconds"] / 60
        return (
            f"{s['done']}/{s['total']} ({pct:.1f}%) "
            f"| {s['rate_per_sec']:.1f} img/s "
            f"| ETA {eta_min:.1f} min"
        )

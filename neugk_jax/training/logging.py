"""Rank-0-only logging helpers (wandb optional)."""

from __future__ import annotations

import os
from typing import Any


class Logger:
    """Tiny wandb wrapper that's a no-op on non-rank-0 processes."""

    def __init__(self, *, is_rank0: bool, cfg: dict | None = None, mode: str = "online"):
        self.is_rank0 = is_rank0
        self.run = None
        if not is_rank0 or mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            print("wandb not installed; logging to stdout only")
            return
        self.run = wandb.init(
            project=(cfg or {}).get("project", "neugk-jax"),
            entity=(cfg or {}).get("entity"),
            name=(cfg or {}).get("run_id"),
            mode=mode,
            config=cfg,
        )

    def log(self, data: dict[str, Any], step: int | None = None, commit: bool = True) -> None:
        if not self.is_rank0:
            return
        if self.run is not None:
            self.run.log(data, step=step, commit=commit)
        else:
            kv = " ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in data.items())
            print(f"[step={step}] {kv}")

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

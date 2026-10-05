"""Rank-0-only logging helpers (wandb optional)."""

from __future__ import annotations

from typing import Any


class Logger:
    """wandb wrapper that logs the full resolved config; a no-op on non-rank-0 processes.

    ``logging`` holds the run settings (``mode``, ``project``, ``entity``, ``run_id``);
    ``mode`` defaults to disabled when the section is missing.
    """

    def __init__(self, *, is_rank0: bool, config: dict | None = None, logging: dict | None = None):
        self.is_rank0 = is_rank0
        self.run = None
        logging = logging or {}
        mode = logging.get("mode", "disabled")
        if not is_rank0 or mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            print("wandb not installed; logging to stdout only")
            return
        self.run = wandb.init(
            project=logging.get("project", "neugk-jax"),
            entity=logging.get("entity"),
            name=logging.get("run_id"),
            mode=mode,
            config=config,
        )

    def log(self, data: dict[str, Any], step: int | None = None, commit: bool = True) -> None:
        if not self.is_rank0:
            return
        if self.run is not None:
            self.run.log(data, step=step, commit=commit)
        else:
            kv = " ".join(
                f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in data.items()
                if isinstance(v, int | float | str)
            )
            if kv:
                print(f"[step={step}] {kv}")

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

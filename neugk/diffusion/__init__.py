"""Diffusion runners and utilities."""

from neugk.diffusion.run import (
    DDPMRunner,
    StudentTRunner,
    FlowMatchingRunner,
    EDMRunner,
    JiTRunner,
)


def get_diffusion_runner(rank: int, cfg, world_size: int):
    """Factory function to get the appropriate diffusion runner based on configuration."""
    diff_cfg = cfg.model.get("diffusion", {})
    noise_dist = diff_cfg.get("noise_distribution", "gaussian").lower()
    formulation = diff_cfg.get("formulation", "ddpm").lower()

    if "ddpm" in formulation:
        assert noise_dist in ["gaussian", "student_t"]
        if noise_dist == "gaussian":
            return DDPMRunner(rank, cfg, world_size)
        elif noise_dist == "student_t":
            return StudentTRunner(rank, cfg, world_size)
    elif "flow" in formulation:
        return FlowMatchingRunner(rank, cfg, world_size)
    elif "edm" in formulation or "karras" in formulation:
        return EDMRunner(rank, cfg, world_size)
    elif "jit" in formulation:
        return JiTRunner(rank, cfg, world_size)


__all__ = ["DDPMRunner", "StudentTRunner", "JiTRunner", "get_diffusion_runner"]

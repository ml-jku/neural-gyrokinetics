from neugk.diffusion.run import (
    DDPMRunner,
    StudentTRunner,
    FlowMatchingRunner,
    EDMRunner,
)


def get_diffusion_runner(rank: int, cfg, world_size: int):
    noise_dist = getattr(cfg.model.diffusion, "noise_distribution", "gaussian").lower()
    formulation = getattr(cfg.model.diffusion, "formulation", "ddpm").lower()

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


__all__ = ["DDPMRunner", "StudentTRunner", "get_diffusion_runner"]

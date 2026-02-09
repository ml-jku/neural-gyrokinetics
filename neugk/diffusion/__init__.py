from neugk.diffusion.run import DDPMRunner, StudentTRunner, LaplaceFlowRunner


def get_diffusion_runner(rank: int, cfg, world_size: int):
    noise_dist = getattr(cfg.model, "noise_distribution", "gaussian").lower()

    if noise_dist == "gaussian":
        return DDPMRunner(rank, cfg, world_size)
    if noise_dist == "student_t":
        return StudentTRunner(rank, cfg, world_size)
    if noise_dist == "laplace":
        return LaplaceFlowRunner(rank, cfg, world_size)


__all__ = ["DDPMRunner", "StudentTRunner", "get_diffusion_runner"]

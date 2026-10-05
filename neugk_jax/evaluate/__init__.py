from neugk_jax.evaluate.base import BaseEvaluator

__all__ = ["BaseEvaluator", "AEEvaluator", "DiffusionEvaluator", "GyroSwinEvaluator"]


def __getattr__(name):
    # lazy re-exports to avoid circular imports with the workflow evaluators
    if name == "AEEvaluator":
        from neugk_jax.pinc.eval import AEEvaluator

        return AEEvaluator
    if name == "DiffusionEvaluator":
        from neugk_jax.diffusion.eval import DiffusionEvaluator

        return DiffusionEvaluator
    if name == "GyroSwinEvaluator":
        from neugk_jax.gyroswin.eval import GyroSwinEvaluator

        return GyroSwinEvaluator
    raise AttributeError(name)

from neugk_jax.evaluate.base import BaseEvaluator, validation_metrics
from neugk_jax.evaluate.integrals import compute_integrals

__all__ = [
    "BaseEvaluator",
    "validation_metrics",
    "AEEvaluator",
    "DiffusionEvaluator",
    "compute_integrals",
]


def __getattr__(name):
    # lazy re-exports to avoid circular imports — workflow evaluators import
    # ``neugk_jax.evaluate.base`` themselves, so we can't eagerly load them here.
    if name == "AEEvaluator":
        from neugk_jax.autoencoders.eval import AEEvaluator
        return AEEvaluator
    if name == "DiffusionEvaluator":
        from neugk_jax.diffusion.eval import DiffusionEvaluator
        return DiffusionEvaluator
    raise AttributeError(name)

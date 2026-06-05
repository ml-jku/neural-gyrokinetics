"""Scalable PINC compression evaluation.

Runs every baseline (GT, traditional ZFP/Wavelet/PCA/JPEG2000/SZ3, neural
fields, autoencoders/VQ-VAE/VAPOR) through one metrics path and stores pickled
results, so the notebooks only load and plot.
"""

from neugk.pinc.eval.reconstructors import (
    Reconstructor,
    GroundTruth,
    Traditional,
    NeuralField,
    Autoencoder,
    traditional_suite,
    nf_scaling_reconstructors,
    traditional_scaling_reconstructors,
)
from neugk.pinc.eval.runner import evaluate_method, run_eval, run_scaling

__all__ = [
    "Reconstructor",
    "GroundTruth",
    "Traditional",
    "NeuralField",
    "Autoencoder",
    "traditional_suite",
    "nf_scaling_reconstructors",
    "traditional_scaling_reconstructors",
    "evaluate_method",
    "run_eval",
    "run_scaling",
]

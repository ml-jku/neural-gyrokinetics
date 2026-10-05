"""External artifact locations, resolved from the environment.

None of these live in the repo: the gyroflow checkpoints, the training
normalization stats, the GKW dataset and the TORAX cases are staged
separately. Each getter raises if its variable is unset, so a missing stage
fails with the variable name rather than a confusing downstream error.
"""

import os

_VARS = {
    "ckpt_dir": ("GYROFLOW_CKPT_DIR", "gyroflow autoencoder + DiT checkpoints"),
    "assets": ("GYROFLOW_ASSETS", "trained surrogate weights"),
    "df_stats": ("GYROFLOW_DF_STATS", "per-mu z-score of the training df"),
    "gkw_raw_dir": ("GKW_RAW_DIR", "GKW K-dumps"),
    "preprocessed_dir": ("PREPROCESSED_DIR", "preprocessed training bins"),
    "dataset_dir": ("GYROFLOW_DATASET_DIR", "condition/flux trajectory set"),
    "case_dir": ("TORAX_CASE_DIR", "TORAX reference case"),
}


def get(name: str) -> str:
    """Resolve one staged location, raising if its variable is unset."""
    try:
        var, what = _VARS[name]
    except KeyError as exc:
        raise KeyError(f"unknown path {name!r}, expected one of {sorted(_VARS)}") from exc
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"set {var} to the {what}")
    return value


def __getattr__(name):
    if name in _VARS:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

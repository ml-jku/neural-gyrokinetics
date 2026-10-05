"""Shared fixtures: public Hugging Face assets (downloaded into the standard HF cache)."""

from __future__ import annotations

import pytest

DATA_REPO = "ml-jku/gyroswin_cbc_id_ood"
LARGE_REPO = "ml-jku/gyroswin_large"
SAMPLE = "iteration_8"


def hf_file(repo_id: str, filename: str, repo_type: str | None = None) -> str:
    """Local path of a Hugging Face file; skips the test without hub access."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        pytest.skip("needs huggingface_hub (pip install huggingface_hub)")
    try:
        return hf_hub_download(repo_id, filename, repo_type=repo_type)
    except Exception as e:
        pytest.skip(f"needs Hugging Face access to {repo_id}/{filename}: {type(e).__name__}: {e}")


@pytest.fixture(scope="session")
def hf_sample():
    """Public CBC snapshot ``iteration_8.h5``: ``root`` dir, ``name``, ``h5`` and ``stats`` paths."""
    import os
    from types import SimpleNamespace

    pytest.importorskip("h5py")
    h5 = hf_file(DATA_REPO, f"preprocessed/{SAMPLE}.h5", repo_type="dataset")
    stats = hf_file(DATA_REPO, "normalization_stats.pkl", repo_type="dataset")
    return SimpleNamespace(root=os.path.dirname(h5), name=SAMPLE, h5=h5, stats=stats)


@pytest.fixture(scope="session")
def hf_large_weights():
    return hf_file(LARGE_REPO, "pytorch_model.bin")

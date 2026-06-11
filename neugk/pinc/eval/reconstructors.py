"""Per-baseline reconstructors for the PINC evaluation.

Each reconstructor turns a trajectory + timesteps into a list of reconstructed
distribution-function snapshots and a compressed size (bytes), so the runner can
score every method through one metrics path. Ported/structured from
`notebooks/01_pinc_evaluation_hf.ipynb`.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

from neugk.pinc.neural_fields.data import CycloneNFDataset
from neugk.pinc.neural_fields.nf_utils import sample_field, load_nf, compress_weights


class Reconstructor(ABC):
    name: str

    @abstractmethod
    def reconstruct(
        self, traj: str, timesteps: Sequence[int], gt: CycloneNFDataset, device: str
    ) -> Tuple[List[torch.Tensor], Optional[int]]:
        """Return (reconstructed dfs per timestep, total compressed bytes or None)."""
        raise NotImplementedError


class GroundTruth(Reconstructor):
    name = "GT"

    def reconstruct(self, traj, timesteps, gt, device):
        dfs = [
            gt.full_df[:, t] if gt.ndim > 5 else gt.full_df
            for t in range(len(timesteps))
        ]
        return dfs, None


class Traditional(Reconstructor):
    """ZFP / Wavelet / PCA / JPEG2000 / SZ3 — `fn(df) -> (recon, _, n_bytes)`."""

    def __init__(self, name: str, fn: Callable):
        self.name = name
        self.fn = fn

    def reconstruct(self, traj, timesteps, gt, device):
        dfs, size = [], 0
        for t in range(len(timesteps)):
            df = gt.full_df[:, t] if gt.ndim > 5 else gt.full_df
            recon, _, cs = self.fn(df)
            dfs.append(recon)
            size += int(cs)
        return dfs, size


class NeuralField(Reconstructor):
    """Per-snapshot neural fields, optionally hybrid-compressed in weight space."""

    def __init__(
        self,
        name: str,
        weights: Dict[str, Dict[int, str]],
        hybrid: Optional[str] = None,
        path: str = None,
        backend: str = "gds",
    ):
        self.name = name
        self.weights = weights  # weights[traj][timestep] -> checkpoint path
        self.hybrid = hybrid  # None | "zfp" | "zipnn"
        self.path = path
        self.backend = backend

    def reconstruct(self, traj, timesteps, gt, device):
        dfs, size = [], 0
        traj_key = traj.replace(".h5", "")
        for i, t in enumerate(timesteps):
            ckpt = self.weights[traj_key][t]
            # build the dataset first so the field grid_size is known: discrete
            # embeddings size their tables from it and load_nf needs it to rebuild.
            nf_data = CycloneNFDataset(
                traj_key,
                timesteps=t,
                path=self.path,
                backend=self.backend,
                realpotens=True,
                normalize="zscore",
                normalize_coords=False,
            )
            nf = load_nf(ckpt, device, grid_size=nf_data.grid_size).to(device)
            if self.hybrid == "zfp":
                nf, _, nbytes = compress_weights(nf, method="zfp", tolerance=1e-3)
                size += nbytes
            elif self.hybrid == "zipnn":
                nf, _, nbytes = compress_weights(nf, method="zipnn")
                size += nbytes
            else:
                size += sum(p.nbytes for p in nf.parameters())
            dfs.append(sample_field(nf, nf_data, device).cpu())
            torch.cuda.empty_cache()
        return dfs, size


class Autoencoder(Reconstructor):
    """Generalizing AE / VQ-VAE (and VAPOR), with per-trajectory val normalization.

    `make_val_dataset(traj)` must return a CycloneAEDataset for the trajectory
    sharing the training normalization stats; the runner wires it from the
    checkpoint config.
    """

    def __init__(
        self,
        name: str,
        model,
        make_val_dataset: Callable,
        vqvae: bool = False,
        vapor: bool = False,
    ):
        self.name = name
        self.model = model
        self.make_val_dataset = make_val_dataset
        self.vqvae = vqvae
        self.vapor = vapor

    def reconstruct(self, traj, timesteps, gt, device):
        ae = self.model.to(device)
        val = self.make_val_dataset(traj)
        dfs, size = [], 0
        for t in timesteps:
            sample = val[t]
            df = sample.df.unsqueeze(0).to(device)
            cond = sample.conditioning.unsqueeze(0).to(device)
            if self.vapor and getattr(val, "separate_zf", False):
                df = df[:, [0, 1]] + df[:, [2, 3]]
            out = ae(df, condition=cond)
            ae_df = out["df"].cpu().squeeze(0)
            if self.vapor and getattr(val, "separate_zf", False):
                zf = ae_df.mean(dim=-1, keepdim=True).expand_as(ae_df)
                ae_df = torch.cat([zf, ae_df - zf], dim=0)
            ae_df = val.denormalize(0, df=ae_df)
            if ae_df.shape[0] == 4:
                ae_df = ae_df[[0, 1]] + ae_df[[2, 3]]
            dfs.append(ae_df)
            if self.vqvae:
                size += out["vq_indices"].to(torch.int16).nbytes
            elif self.vapor:
                size += int(gt.full_df.numel() * 2 * 4 / 64)  # 2ch f32, hardcoded ratio
            else:
                # latent is stored at bf16 (2 bytes/elem), not the f32 the encoder returns
                size += ae.encode(df, condition=cond)[0].numel() * 2
            torch.cuda.empty_cache()
        return dfs, size


def traditional_suite(
    error_args: Optional[Dict[str, dict]] = None,
) -> List[Traditional]:
    """Build the traditional baselines (skips SZ3 if `pysz` is unavailable)."""
    from neugk.pinc.eval import trad

    error_args = error_args or {}
    suite = []
    for name, fn in [
        ("ZFP", trad.zfp_recon),
        ("Wavelet", trad.wavelet_recon),
        ("PCA", trad.pca_recon),
        ("JPEG2000", trad.jpeg2000_recon),
    ]:
        suite.append(Traditional(name, fn))
    if hasattr(trad, "sz3_recon"):
        try:
            import pysz  # noqa: F401

            suite.append(Traditional("SZ3", trad.sz3_recon))
        except Exception:
            pass
    return suite


# ---------------------------------------------------------------------------
# Scaling reconstructor builders (for run_scaling rate-distortion curves)
# ---------------------------------------------------------------------------


def nf_scaling_reconstructors(
    ckp_dir: str,
    device: str = "cuda",
    model_type: str = "mlp",
    hybrid: Optional[str] = None,
    include_int_only: bool = False,
    min_cr: int = 0,
    max_cr: int = 10_000_000,
    path: str = "/local00/bioinf/galletti/preprocessed_kvikio",
    backend: str = "gds",
) -> List[Tuple[int, NeuralField]]:
    """Scan ``ckp_dir`` for NF checkpoints and build one NeuralField reconstructor per CR.

    Expects files named ``{model_type}_{traj}_t{timestep}_x{cr}.pt`` (pre-training)
    and ``int_{model_type}_{traj}_t{timestep}_x{cr}.pt`` (PINC fine-tuned).
    When ``include_int_only``, only the PINC-fine-tuned checkpoints are used.
    The ``hybrid`` parameter enables weight-space hybrid compression (``"zfp"`` or ``"zipnn"``).

    Returns a list of ``(cr, NeuralField)`` sorted by CR, suitable for wrapping in a
    ``run_scaling`` group.
    """
    import re
    from pathlib import Path

    ckp_dir = Path(ckp_dir)
    # {cr: {traj: {t: ckpt_path}}}
    weights_by_cr: Dict[int, Dict[str, Dict[int, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # pre-training checkpoints
    pretrain_pat = re.compile(rf"{model_type}_([\w.]+)_t(\d+)_x(\d+)\.pt$")
    for f in ckp_dir.glob(f"{model_type}_*.pt"):
        m = pretrain_pat.match(f.name)
        if not m:
            continue
        traj, t, cr = m.groups()
        traj = traj.split(".")[0]
        t, cr = int(t), int(cr)
        if cr < min_cr or cr > max_cr:
            continue
        weights_by_cr[cr][traj][t] = f.as_posix()

    # PINC-fine-tuned checkpoints: ONLY for the PINC family. The plain-NF family
    # keeps the density pretrain (mlp_*); otherwise the int_mlp_* overwrite would
    # collapse both families onto the same PINC checkpoints (identical curves).
    if include_int_only:
        int_pat = re.compile(rf"int_{model_type}_([\w.]+)_t(\d+)_x(\d+)\.pt$")
        has_int: set = set()
        for f in ckp_dir.glob(f"int_{model_type}_*.pt"):
            m = int_pat.match(f.name)
            if not m:
                continue
            traj, t, cr = m.groups()
            traj = traj.split(".")[0]
            t, cr = int(t), int(cr)
            if cr < min_cr or cr > max_cr:
                continue
            weights_by_cr[cr][traj][t] = f.as_posix()
            has_int.add(cr)
        weights_by_cr = {cr: w for cr, w in weights_by_cr.items() if cr in has_int}

    # build reconstructors
    results = []
    for cr in sorted(weights_by_cr.keys()):
        name = f"{'NF+ZFP' if hybrid else 'NF'}_x{cr}"
        r = NeuralField(
            name=name,
            weights=dict(weights_by_cr[cr]),
            hybrid=hybrid,
            path=path,
            backend=backend,
        )
        results.append((cr, r))
    return sorted(results, key=lambda x: x[0])


def traditional_scaling_reconstructors(
    method: str,
    param_values: Sequence,
    param_name: str = "tolerance",
) -> List[Traditional]:
    """Build Traditional reconstructors at multiple compression settings.

    ``method`` is one of ``"ZFP"``, ``"Wavelet"``, ``"PCA"``, ``"JPEG2000"``.
    ``param_values`` is a list of parameter values (tolerance, threshold, n_components, quality).
    Returns one ``Traditional`` reconstructor per parameter value, named ``ZFP_tol{N}`` etc.
    """
    from functools import partial
    from neugk.pinc.eval import trad

    fn_map = {
        "ZFP": trad.zfp_recon,
        "Wavelet": trad.wavelet_recon,
        "PCA": trad.pca_recon,
        "JPEG2000": trad.jpeg2000_recon,
    }
    if method not in fn_map:
        raise ValueError(
            f"Unknown traditional method: {method}. Choose from {list(fn_map)}."
        )

    base_fn = fn_map[method]
    results = []
    for val in param_values:
        fn = partial(base_fn, **{param_name: val})
        results.append(Traditional(f"{method}_{param_name[:3]}{val}", fn))
    return results

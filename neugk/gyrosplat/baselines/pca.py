"""Per-snapshot PCA of the (velocity x parallel) by (x x y) unfolding of each stored real
channel: centred, with the mean, the components, the scores and the variances stored, at
the rank whose size is closest to the budget (the evaluation of PINC)."""
from __future__ import annotations

import math
from typing import List

import torch


def pca_channels(backend, real_fields: List[torch.Tensor], budget_reals: int):
    """Centred PCA of every real channel.  Returns (H_hat, achieved reals per species, {"rank": r})."""
    fields, achieved = [], []
    for f in real_fields:
        m = f.shape[1] * f.shape[2] * f.shape[3]
        n = f.shape[4] * f.shape[5]

        def stored(r):  # scores, components, mean and variances of both channels
            return 2 * (r * (m + n) + n + r)

        r = min(
            range(1, min(m, n) + 1),
            key=lambda k: abs(math.log(stored(k) / budget_reals)),
        )
        rec = []
        for ch in f:
            X = ch.reshape(m, n).to(torch.float64)
            mean = X.mean(0, keepdim=True)
            _, V = torch.linalg.eigh((X - mean).T @ (X - mean))  # ascending eigenvalues
            Vr = V[:, -r:]
            rec.append(((X - mean) @ Vr @ Vr.T + mean).reshape(ch.shape))
        fields.append(torch.stack(rec, 0).to(torch.float32))
        achieved.append(stored(r))
    with torch.no_grad():
        H_hat = backend.to_spectral(fields)
    return H_hat, achieved, {"rank": r}

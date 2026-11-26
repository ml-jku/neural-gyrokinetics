"""`python -m neugk.gyrosplat table results/*.json`: medians of this package's result files."""
from __future__ import annotations

from typing import Sequence

from neugk.gyrosplat.eval.paper.rows import grouped, medians, package_rows


def print_table(paths: Sequence[str]) -> None:
    rows = package_rows(paths)
    print(
        "| tier | method / stage | n | CR | PSNR_f | PSNR_phi | flux relL1 | kappa/kappa_truth |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for key, group in sorted(
        grouped(rows).items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
    ):
        m = medians(group)
        name = key[1] if isinstance(key[1], str) else f"{key[1][0]} (stage {key[1][1]})"

        def f(k, d):
            v = m.get(k)
            return "—" if v is None else f"{v:.{d}f}"

        print(
            f"| {key[0]} | {name} | {m['n']} | {f('CR', 1)} | {f('psnr_f', 2)} | {f('psnr_phi', 2)} "
            f"| {f('flux_rel_l1', 4)} | {f('kappa_rel', 3)} |"
        )

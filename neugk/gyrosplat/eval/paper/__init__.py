"""Tables and figures of the paper from per-snapshot result rows.

    from neugk.gyrosplat.eval.paper import load_rows, load_experiments, main_table, ladder_table

`rows` defines the row format and converts this package's result files into it; `tables`
builds the tables of the single-seed rows (the ablation ladder), `experiments` the tables of
the byte-matched comparison, the fitting seeds and the appendix experiments, and `figures`
the figures.  matplotlib is only needed for the figures (`pip install -e ".[paper]"`).
"""
from neugk.gyrosplat.eval.paper.rows import (
    DISPERSION_ORDER,
    LADDER,
    LADDER_MAIN,
    METHODS,
    PACKAGE_METHODS,
    SNAPSHOT_CLASSES,
    TABLE1_ORDER,
    load_rows,
    means,
    medians,
    package_rows,
    quantile,
    rows_of,
    select,
    sign_flips,
)
from neugk.gyrosplat.eval.paper.tables import (
    compare_table,
    dispersion_table,
    families_table,
    ladder_table,
    per_class_table,
    storage_table,
    transfer_table,
    wavelet_table,
)
from neugk.gyrosplat.eval.paper.experiments import (
    bytes_table,
    damping_table,
    db4_table,
    final_fit_table,
    heldout_table,
    latent_steps_table,
    latent_table,
    load_experiments,
    main_table,
    paired_gain,
    rd_budget_table,
    rd_table,
    refinement_table,
    regime_table,
    relative_error_table,
    restart_table,
    seed_dispersion_table,
    stage_table,
    timing_table,
    views_table,
)


def scatter_figure(*args, **kwargs):
    from neugk.gyrosplat.eval.paper.figures import scatter_figure as f

    return f(*args, **kwargs)


def flux_spectrum_figure(*args, **kwargs):
    from neugk.gyrosplat.eval.paper.figures import flux_spectrum_figure as f

    return f(*args, **kwargs)


def rd_figure(*args, **kwargs):
    from neugk.gyrosplat.eval.paper.figures import rd_figure as f

    return f(*args, **kwargs)


__all__ = [
    "bytes_table",
    "damping_table",
    "db4_table",
    "final_fit_table",
    "heldout_table",
    "latent_steps_table",
    "latent_table",
    "load_experiments",
    "main_table",
    "paired_gain",
    "rd_budget_table",
    "rd_figure",
    "rd_table",
    "refinement_table",
    "regime_table",
    "relative_error_table",
    "restart_table",
    "seed_dispersion_table",
    "stage_table",
    "timing_table",
    "views_table",
    "DISPERSION_ORDER",
    "LADDER",
    "LADDER_MAIN",
    "METHODS",
    "PACKAGE_METHODS",
    "SNAPSHOT_CLASSES",
    "TABLE1_ORDER",
    "compare_table",
    "dispersion_table",
    "families_table",
    "flux_spectrum_figure",
    "ladder_table",
    "load_rows",
    "means",
    "medians",
    "package_rows",
    "per_class_table",
    "quantile",
    "rows_of",
    "scatter_figure",
    "select",
    "sign_flips",
    "storage_table",
    "transfer_table",
    "wavelet_table",
]

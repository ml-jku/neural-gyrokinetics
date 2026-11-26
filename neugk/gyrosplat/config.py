"""Recipe configuration.

All knobs of the codec are plain dataclasses so a recipe can be written as a YAML file,
loaded with `load_config`, and overridden from the command line (`key.subkey=value`).
The defaults reproduce the historical GKW 302x preset (corrected CR ~189x); the YAML files under
`configs/gyrosplat/recipe/` hold both tiers and the small synthetic recipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from omegaconf import OmegaConf


@dataclass
class InitConfig:
    """How the base atoms are placed before the descent.

    "field": centres drawn with probability ~ |f|^2, constant widths.
    "random": centres uniform over the domain (the textbook start).
    """

    method: str = "field"
    sigma_cells: float = 2.0  # initial width, in grid cells, on every axis


@dataclass
class FitConfig:
    """The AdamW descent on atom geometry and amplitudes against the normalised field."""

    steps: int = 1000
    lr_scale: float = (
        1.0  # multiplies the base learning rates (mu 1e-3, L 5e-3, amps 1e-2)
    )
    loss: str = "huber"  # "huber" or "mse"
    huber_delta: float = 1.0
    chunk: int = 16384  # spatial rows per render block (memory/speed trade-off)
    warm_start: bool = (
        True  # solve the amplitudes in closed form once before the descent
    )
    polish_steps: int = 2  # a few extra steps at low learning rates after the schedule
    schedule: str = "cosine"  # "cosine": per-group rates, cosine decay | "flat": plain Adam at `flat_lr`
    flat_lr: float = 5e-3


@dataclass
class VarProConfig:
    """The closed-form amplitude solve at fixed geometry."""

    ridge: float = 1e-3
    per_column: bool = False  # scale the ridge per Gram diagonal instead of by its mean
    chunk: int = 4096
    after_fit: bool = False  # re-solve the amplitudes after the descent
    potential_weight: float = (
        1.0  # weight of the backend's potential summary in the solve (0: field only)
    )


@dataclass
class PhysicsFitConfig:
    """A short descent on every atom parameter against the backend's physics terms, evaluated
    on the de-normalised field.  The recipe runs 40 epochs with the flux term off (the flux
    is the Gauss-Newton stage's job)."""

    epochs: int = 40
    lr_scale: float = 1.0  # multiplies the base rates (mu 5e-5, L 3e-4, amps 1e-3)
    field_weight: float = 1.0
    potential_weight: float = 1.0
    flux_weight: float = 0.0


@dataclass
class CarrierConfig:
    """Groups of `tied_k` ky carriers initialized with equal envelopes at the flux tail.
    Envelopes are independent by default; share_envelopes enforces one geometry
    parameter row per group throughout fitting and storage."""

    kind: str = "tied_gabor"  # "none" | "tied_gabor"
    fraction: float = 0.25  # the carrier share of the budget
    tied_k: int = 7  # carriers per group; equality persists only with share_envelopes
    share_envelopes: bool = (
        False  # one trainable geometry row per carrier group if true
    )
    tail_bins: Optional[
        Tuple[int, ...]
    ] = None  # the ky bins of the tail (None: the upper half of the band)
    sigma_y_cells: float = 1.5  # envelope width along the carrier axis, in grid cells
    residual_solve: bool = (
        True  # solve only the carrier amplitudes, against the residual of the base
    )
    flux_gate: float = (
        5.0  # leave the carriers inert when the band flux magnitude is below this
    )
    cofit: str = "routed"  # "routed": the co-fit below | "none"
    routed_steps: int = (
        60  # routed co-fit: epochs (physics -> base rows, flux -> carrier rows)
    )
    routed_steps_hard: int = 120  # ... when the potential is placement-bound (gradient ratio above hard_ratio)
    hard_ratio: float = 50.0
    routed_flux_weight: float = 1.0


@dataclass
class GaussNewtonConfig:
    """The physics stage: Gauss-Newton on the carrier amplitudes against the flux rows."""

    steps: int = 8
    mu: float = 1e-4  # Levenberg damping, relative to the row scale
    carriers_only: bool = True


@dataclass
class NeuralFieldConfig:
    """The neural-field baseline: the density phase of PINC."""

    layers: int = 5
    width: Optional[int] = None  # None: the widest network within the budget
    epochs: int = 20
    batch: int = 4096
    lr: float = 7e-3
    weight_decay: float = 1e-8
    # share of the grid that an epoch draws, from the first epoch to the last
    subsample: List[float] = field(default_factory=lambda: [0.2, 1.0])
    # z-score the channels over the whole grid (None) or per index of one grid axis
    zscore_axis: Optional[int] = None
    compile: bool = True


@dataclass
class BaselineConfig:
    """Choices of the baseline codecs."""

    fourier_all_axes: bool = (
        True  # top-K of the full Fourier transform; False: cells of the spectral object
    )
    wavelet: str = "bior4.4"
    wavelet_level: int = 1
    objective_warmup: int = (
        400  # Adam steps on the backend's physics terms before the Gauss-Newton
    )
    objective_relative_damping: float = 1e-8  # damping of the "+ objective" Gauss-Newton, as a fraction of max diag(J J^T)
    splats_plain: bool = (
        True  # field-only splats as the textbook fit: global z-score, random
    )
    # placement, flat-rate Adam on the MSE, no amplitude solve, 16 reals per atom
    nf: NeuralFieldConfig = field(default_factory=NeuralFieldConfig)


@dataclass
class RecipeConfig:
    """One complete recipe: the atom budget and every stage's knobs."""

    n_equiv: int = 4337  # the budget in atom equivalents, split by carriers.fraction
    normalize: str = "zonal"  # "zonal": separate z-score for the ky = 0 part | "global"
    init: InitConfig = field(default_factory=InitConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    varpro: VarProConfig = field(default_factory=VarProConfig)
    physics: PhysicsFitConfig = field(default_factory=PhysicsFitConfig)
    carriers: CarrierConfig = field(default_factory=CarrierConfig)
    gn: GaussNewtonConfig = field(default_factory=GaussNewtonConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    band: Tuple[int, int] = (1, 16)  # the supervised ky band [lo, hi)
    seed: int = 0
    reals_per_atom: int = (
        17  # storage: 5 centre + 6 + 3 Cholesky + 2 amplitude + 1 carrier wavenumber
    )
    storage_bits: int = (
        32  # 16: round the atom parameters to half precision and count them at 16 bit
    )
    verbose: bool = True


def load_config(
    path: Optional[str] = None, overrides: Optional[List[str]] = None
) -> RecipeConfig:
    """Build a RecipeConfig from the defaults, an optional YAML file and `key=value` overrides."""
    base = OmegaConf.structured(RecipeConfig)
    if path:
        base = OmegaConf.merge(base, OmegaConf.load(path))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(base)


def to_yaml(cfg: RecipeConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))

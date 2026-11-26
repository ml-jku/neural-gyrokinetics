"""Dataset loaders.  Each returns a `Snapshot`; the matching backend does the physics."""
from neugk.gyrosplat.datasets.synthetic import make_synthetic_snapshot

__all__ = ["make_synthetic_snapshot"]

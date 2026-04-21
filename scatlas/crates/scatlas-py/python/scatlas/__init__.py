"""scatlas — Rust-first scRNA-seq atlas computation."""

from scatlas._scatlas_native import __version__
from scatlas import ext, metrics, pp, stats, tl

__all__ = ["__version__", "ext", "metrics", "pp", "stats", "tl"]

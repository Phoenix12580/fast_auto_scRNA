"""Stage 11 — per-cluster ROGUE purity (entropy + LOESS).

``core`` — R-faithful ROGUE port (entropy_table / entropy_fit / se_fun /
calculate_rogue / rogue_per_cluster / _remove_top_outliers).
``score`` — the pipeline-facing ``rogue_mean`` wrapper that drives the
per-(cluster, sample) loop and tolerates single-cell LOESS failures.
"""
from .score import rogue_mean
from .core import (
    entropy_table,
    entropy_fit,
    se_fun,
    calculate_rogue,
    rogue_per_cluster,
    filter_matrix,
    determine_k,
)

__all__ = [
    "rogue_mean",
    "entropy_table",
    "entropy_fit",
    "se_fun",
    "calculate_rogue",
    "rogue_per_cluster",
    "filter_matrix",
    "determine_k",
]

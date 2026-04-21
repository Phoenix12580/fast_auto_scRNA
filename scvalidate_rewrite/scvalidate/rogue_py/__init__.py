"""ROGUE: entropy-based cluster purity.

Port of PaulingLiu/ROGUE (R, master, ``R/ROGUE.R``).

Reference: Liu et al. 2020, Nat Commun.
    "An entropy-based metric for assessing the purity of single cell populations"
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7308400/

Core formula::

    ROGUE = 1 - sum(|ds|_sig) / (sum(|ds|_sig) + K)

where ``ds`` is the entropy reduction of a gene against the S-E loess fit,
summed over genes passing ``p.adj < cutoff AND p.value < cutoff``.

K defaults: 45 for UMI (10x droplet), 500 for full-length (Smart-seq).
"""

from scvalidate.rogue_py.core import (
    entropy_table,
    entropy_fit,
    se_fun,
    calculate_rogue,
    rogue_per_cluster,
    filter_matrix,
    determine_k,
)

__all__ = [
    "entropy_table",
    "entropy_fit",
    "se_fun",
    "calculate_rogue",
    "rogue_per_cluster",
    "filter_matrix",
    "determine_k",
]

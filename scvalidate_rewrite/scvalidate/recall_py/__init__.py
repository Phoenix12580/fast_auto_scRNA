"""recall: knockoff calibration for over-clustering.

Port of lcrawlab/recall (R, main).

Reference: DenAdel & Crawford 2024–2025, AJHG/biorxiv.
    "A knockoff calibration method to avoid over-clustering in scRNA-seq"

Core idea: for each variable gene, sample an artificial "knockoff" gene from
the same null distribution (ZIP or NB). Run standard clustering on the
augmented matrix; for each cluster pair, compute per-gene DE p-values, then
compare -log10(p_real) - -log10(p_knockoff) via the Barber-Candès knockoff
filter. If no pair has any selected real gene, the clustering is
over-partitioned → reduce resolution and re-cluster.

v0.1 status — ZIP and NB parameter estimation are implemented.
Orchestrator (``find_clusters_recall``) is a TODO skeleton.
"""

from scvalidate.recall_py.knockoff import (
    estimate_zip,
    estimate_nb,
    sample_zip,
    sample_nb,
    generate_knockoff_matrix,
    knockoff_threshold_offset1,
)
from scvalidate.recall_py.core import RecallResult, find_clusters_recall

__all__ = [
    "RecallResult",
    "estimate_zip",
    "estimate_nb",
    "sample_zip",
    "sample_nb",
    "generate_knockoff_matrix",
    "knockoff_threshold_offset1",
    "find_clusters_recall",
]

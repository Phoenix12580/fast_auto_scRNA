"""Pipeline-facing ROGUE aggregation.

Drives the per-(cluster, sample) loop, tolerating LOESS numerical
failures so a single degenerate subset only invalidates that one cell,
not the whole route.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def rogue_mean(
    counts_gxc,
    cluster_labels: np.ndarray,
    sample_labels: np.ndarray | None = None,
    *,
    platform: str = "UMI",
    min_cell_n: int = 10,
    gene_names: list[str] | None = None,
) -> dict:
    """ROGUE mean + per-cluster map.

    Parameters
    ----------
    counts_gxc
        Raw UMI counts, ``(n_genes, n_cells)`` sparse CSR or dense.
    cluster_labels
        Per-cell cluster assignment.
    sample_labels
        Per-cell sample / batch ID. None → single "_pooled" sample.
    platform
        "UMI" (default) or "full-length".

    Returns
    -------
    dict with keys ``mean``, ``median``, ``per_cluster``,
    ``n_clusters_scored``, ``n_skipped``, ``n_pair_failures``.
    """
    from .core import se_fun, calculate_rogue, _remove_top_outliers

    if sample_labels is None:
        sample_labels = np.full(len(cluster_labels), "_pooled", dtype=object)
    labels = np.asarray(list(cluster_labels))
    samples = np.asarray(list(sample_labels))
    unique_clusters = pd.unique(labels)
    unique_samples = pd.unique(samples)

    matrix = pd.DataFrame(
        np.full((len(unique_samples), len(unique_clusters)), np.nan),
        index=unique_samples, columns=unique_clusters,
    )
    n_failed = 0
    for cluster in unique_clusters:
        for sample in unique_samples:
            sel = (labels == cluster) & (samples == sample)
            if int(sel.sum()) < min_cell_n:
                continue
            if sp.issparse(counts_gxc):
                sub_expr = counts_gxc[:, np.where(sel)[0]]
            else:
                sub_expr = counts_gxc[:, sel]
            try:
                se = se_fun(sub_expr, span=0.5, r=1.0, mt_method="fdr_bh",
                            gene_names=gene_names)
                se = _remove_top_outliers(se, sub_expr, n=2, span=0.5, r=1.0,
                                          mt_method="fdr_bh")
                matrix.loc[sample, cluster] = calculate_rogue(se, platform=platform)
            except Exception:
                n_failed += 1

    values = matrix.to_numpy(dtype=float)
    flat = values[~np.isnan(values)]

    per_cluster: dict = {}
    for cluster_id in matrix.columns:
        col_vals = matrix[cluster_id].dropna().to_numpy(dtype=float)
        if len(col_vals) > 0:
            per_cluster[str(cluster_id)] = float(np.mean(col_vals))

    return {
        "mean": float(np.mean(flat)) if len(flat) else float("nan"),
        "median": float(np.median(flat)) if len(flat) else float("nan"),
        "per_cluster": per_cluster,
        "n_clusters_scored": len(per_cluster),
        "n_skipped": int(matrix.shape[1]) - len(per_cluster),
        "n_pair_failures": int(n_failed),
    }

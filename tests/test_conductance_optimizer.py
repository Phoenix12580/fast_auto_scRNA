"""Conductance-based resolution optimizer tests.

1. mean_conductance sanity: a perfect 2-block graph with zero inter-block
   edges has conductance 0; a random graph has conductance ~0.5.
2. optimize_resolution_conductance picks the low-resolution (few-cluster)
   partition on a well-separated 2-blob synthetic dataset, NOT the
   over-fragmented high-resolution one. This is the bug the old
   graph_silhouette optimizer had.
"""
from __future__ import annotations

import numpy as np
import pytest


def _build_block_graph(n_per_block=100, n_blocks=2, within_p=0.3, between_p=0.0, seed=0):
    """Stochastic block model — clean two-community graph."""
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    rows, cols, vals = [], [], []
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            p = within_p if bi == bj else between_p
            if p == 0.0:
                continue
            for i in range(bi * n_per_block, (bi + 1) * n_per_block):
                for j in range(bj * n_per_block, (bj + 1) * n_per_block):
                    if i < j and rng.random() < p:
                        rows.append(i); cols.append(j); vals.append(1.0)
                        rows.append(j); cols.append(i); vals.append(1.0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def test_mean_conductance_perfect_split():
    from fast_auto_scrna.cluster.resolution import mean_conductance
    G = _build_block_graph(n_per_block=50, between_p=0.0, within_p=0.3, seed=0)
    labels = np.array([0] * 50 + [1] * 50)
    assert mean_conductance(G, labels) == 0.0


def test_mean_conductance_worst_split():
    """Labels that chop each block in half → many boundary edges."""
    from fast_auto_scrna.cluster.resolution import mean_conductance
    G = _build_block_graph(n_per_block=50, between_p=0.0, within_p=0.3, seed=0)
    # Interleaving labels intentionally creates cross-block partitions
    # that go against the natural structure.
    bad = np.array([i % 2 for i in range(100)])
    good = np.array([0] * 50 + [1] * 50)
    assert mean_conductance(G, bad) > mean_conductance(G, good)


def test_optimize_resolution_conductance_picks_two_cluster_solution():
    """On a clean two-blob graph the optimizer should pick the resolution
    that gives k=2, not a fragmented high-k partition."""
    import anndata as ad
    import scipy.sparse as sp
    import scanpy as sc
    from fast_auto_scrna.cluster.resolution import (
        optimize_resolution_conductance,
        pick_best_resolution,
    )

    # Build a synthetic embedding with 2 well-separated blobs.
    rng = np.random.default_rng(42)
    n = 400
    X1 = rng.standard_normal((n // 2, 10)).astype(np.float32)
    X2 = rng.standard_normal((n // 2, 10)).astype(np.float32) + 6.0
    X = np.vstack([X1, X2])
    a = ad.AnnData(X=X)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X")
    G = a.obsp["connectivities"].tocsr()
    a.obsp["bbknn_connectivities"] = G  # optimize_resolution_conductance expects this

    resolutions = [0.05, 0.2, 0.5, 1.0, 2.0]
    curve = optimize_resolution_conductance(
        a, method="bbknn",
        resolutions=resolutions,
        seed=0, leiden_n_iterations=2, verbose=False,
    )

    # There must be a resolution yielding exactly k=2.
    k2_rows = curve[curve["n_clusters"] == 2]
    assert len(k2_rows) >= 1, f"no k=2 result: {curve}"

    best_r = pick_best_resolution(
        curve, metric="conductance", direction="min",
    )
    row = curve.loc[curve["resolution"] == best_r].iloc[0]
    # Must land on k=2 (tight blobs) — not a fragmented high-k solution.
    assert int(row["n_clusters"]) == 2, (
        f"conductance picker chose k={int(row['n_clusters'])}, expected 2.\n{curve}"
    )

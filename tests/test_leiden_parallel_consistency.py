"""Parallel-vs-sequential Leiden sweep parity test.

GS-4 swaps the resolution-sweep in ``optimize_resolution_graph_silhouette``
from a serial loop to a ProcessPoolExecutor. leidenalg is deterministic
under a fixed ``random_state``, so parallel output MUST match sequential
bit-for-bit. This test locks that contract in.
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_two_cluster_graph(n: int = 800, seed: int = 42):
    """Small well-separated 2-blob knn graph for clustering tests."""
    import scanpy as sc
    import anndata as ad

    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal((n // 2, 20)).astype(np.float32)
    X2 = rng.standard_normal((n // 2, 20)).astype(np.float32) + 5.0
    X = np.vstack([X1, X2])
    a = ad.AnnData(X=X)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X")
    return a.obsp["connectivities"].tocsr()


@pytest.mark.parametrize("resolutions", [
    [0.1, 0.3, 0.5],
    [0.05, 0.2, 0.4, 0.8, 1.2],
])
def test_leiden_sweep_parallel_matches_sequential(resolutions):
    import scanpy as sc
    import anndata as ad
    import scipy.sparse as sp
    from fast_auto_scrna.cluster.resolution import _leiden_sweep

    G = _make_two_cluster_graph(n=800)
    n = G.shape[0]

    # Sequential reference
    a = ad.AnnData(X=sp.csr_matrix((n, 1)))
    ref: dict[float, np.ndarray] = {}
    for r in resolutions:
        sc.tl.leiden(
            a, resolution=r, key_added=f"serial_r{r}", adjacency=G,
            flavor="igraph", n_iterations=2,
            directed=False, random_state=0,
        )
        ref[r] = a.obs[f"serial_r{r}"].astype(int).to_numpy()

    # Parallel via ProcessPoolExecutor
    par = _leiden_sweep(G, list(resolutions), seed=0, n_iterations=2)

    for r in resolutions:
        assert np.array_equal(ref[r], par[r]), (
            f"leiden parallel/serial mismatch at r={r}: "
            f"serial k={len(np.unique(ref[r]))}, parallel k={len(np.unique(par[r]))}"
        )


def test_leiden_sweep_single_resolution_skips_pool():
    """Single resolution should short-circuit the process pool."""
    import scanpy as sc
    import anndata as ad
    import scipy.sparse as sp
    from fast_auto_scrna.cluster.resolution import _leiden_sweep

    G = _make_two_cluster_graph(n=400)
    n = G.shape[0]

    a = ad.AnnData(X=sp.csr_matrix((n, 1)))
    sc.tl.leiden(
        a, resolution=0.3, key_added="ref", adjacency=G,
        flavor="igraph", n_iterations=2, directed=False, random_state=0,
    )
    ref = a.obs["ref"].astype(int).to_numpy()

    out = _leiden_sweep(G, [0.3], seed=0, n_iterations=2)
    assert set(out.keys()) == {0.3}
    assert np.array_equal(ref, out[0.3])

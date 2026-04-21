"""Smoke + parity tests for scatlas.tl.umap."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


class FakeAnnData:
    def __init__(self, n: int, conn: sp.spmatrix, pca: np.ndarray | None = None):
        self.X = None
        self.n_obs = n
        self.n_vars = 0
        self.obs = type("O", (), {"columns": []})()
        self.obsm = {}
        if pca is not None:
            self.obsm["X_pca"] = pca
        self.obsp = {"bbknn_connectivities": conn}
        self.varm = {}
        self.uns = {}
        self.var = type("V", (), {"columns": []})()

    def copy(self):
        return self  # tests don't rely on copy semantics


def _build_two_cluster_connectivities(n_per: int = 60, seed: int = 0) -> sp.csr_matrix:
    """Fully-connected within-cluster, zero cross-cluster."""
    rng = np.random.default_rng(seed)
    n = n_per * 2
    rows, cols, vals = [], [], []
    for i in range(n):
        cluster = i // n_per
        base = cluster * n_per
        for j in range(n_per):
            other = base + j
            if other != i:
                rows.append(i)
                cols.append(other)
                vals.append(1.0 - 0.1 * rng.random())
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def test_fit_ab_defaults():
    from scatlas.tl import fit_ab

    a, b = fit_ab(0.5, 1.0)
    assert abs(a - 0.583) < 0.03
    assert abs(b - 1.334) < 0.03


def test_umap_separates_two_clusters():
    from scatlas import tl

    conn = _build_two_cluster_connectivities(60, seed=1)
    ad = FakeAnnData(conn.shape[0], conn)
    emb = tl.umap(ad, init="random", random_state=7, n_epochs=200)
    assert emb.shape == (120, 2)

    # Cluster centroids should be well separated
    c0 = emb[:60].mean(axis=0)
    c1 = emb[60:].mean(axis=0)
    sep = float(np.linalg.norm(c0 - c1))

    within0 = float(np.linalg.norm(emb[:60] - c0, axis=1).mean())
    within1 = float(np.linalg.norm(emb[60:] - c1, axis=1).mean())
    within = 0.5 * (within0 + within1)
    assert sep > 2 * within, f"UMAP failed to separate: sep={sep}, within={within}"


def test_umap_pca_init():
    from scatlas import tl

    conn = _build_two_cluster_connectivities(40, seed=2)
    n = conn.shape[0]
    rng = np.random.default_rng(0)
    # Synthetic PCA embedding reflecting the two-cluster structure
    pca = np.vstack([
        rng.normal(loc=[-5, 0], scale=0.5, size=(40, 2)),
        rng.normal(loc=[5, 0], scale=0.5, size=(40, 2)),
    ]).astype(np.float32)
    ad = FakeAnnData(n, conn, pca=pca)
    emb = tl.umap(ad, init="pca", random_state=0, n_epochs=100)
    assert emb.shape == (n, 2)
    # uns records fit results
    assert "a" in ad.uns["umap"]
    assert ad.uns["umap"]["params"]["init"] == "pca"


def test_umap_missing_connectivities_errors():
    from scatlas import tl

    class NoConn:
        n_obs = 10
        obsm = {}
        obsp = {}
        uns = {}

    with pytest.raises(KeyError, match="bbknn_connectivities"):
        tl.umap(NoConn(), init="random")

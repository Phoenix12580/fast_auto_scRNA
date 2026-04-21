"""Parity tests for scatlas.pp.pca vs sklearn.decomposition.TruncatedSVD."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def _principal_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Maximum principal angle (degrees) between column spans of u and v.

    Two rank-k subspaces are close iff all principal angles are small.
    Used to compare PCA outputs across two randomized SVD
    implementations where signs/ordering may flip.
    """
    qu, _ = np.linalg.qr(u)
    qv, _ = np.linalg.qr(v)
    svals = np.linalg.svd(qu.T @ qv, compute_uv=False)
    svals = np.clip(svals, -1.0, 1.0)
    return float(np.degrees(np.arccos(svals.min())))


def test_pca_csr_matches_sklearn_subspace():
    """Embedding subspace of scatlas PCA ≈ sklearn TruncatedSVD on same X.

    Use a low-rank-plus-noise matrix so the top-k subspace is well
    defined (clear singular value gap) — two randomized SVDs with
    different random seeds still agree on the dominant subspace.
    """
    from scatlas import pp

    pytest.importorskip("sklearn")
    from sklearn.decomposition import TruncatedSVD

    rng = np.random.default_rng(0)
    n, g = 400, 50
    rank_true = 8
    U_true = rng.normal(size=(n, rank_true)).astype(np.float32)
    V_true = rng.normal(size=(rank_true, g)).astype(np.float32)
    signal = U_true @ V_true
    noise = 0.05 * rng.normal(size=(n, g)).astype(np.float32)
    dense = signal + noise
    dense[np.abs(dense) < 0.2] = 0.0  # sparsify
    X = sp.csr_matrix(dense.astype(np.float32))

    # scatlas
    class FakeAnnData:
        def __init__(self, X):
            self.X = X
            self.obsm = {}
            self.varm = {}
            self.uns = {}
            self.n_vars = X.shape[1]
            self.var = type("V", (), {"columns": []})()

    ad = FakeAnnData(X)
    out = pp.pca(ad, n_comps=rank_true, n_power_iter=6, random_state=42)
    emb_sc = out["embedding"]

    # sklearn
    svd = TruncatedSVD(
        n_components=rank_true, algorithm="randomized", n_iter=6, random_state=42
    )
    emb_sk = svd.fit_transform(X.astype(np.float32))

    assert emb_sc.shape == emb_sk.shape
    angle = _principal_angle_deg(emb_sc, emb_sk)
    assert angle < 3.0, f"principal angle = {angle:.2f}°, subspaces differ"


def test_pca_csr_singular_values_match_sklearn():
    from scatlas import pp

    pytest.importorskip("sklearn")
    from sklearn.decomposition import TruncatedSVD

    rng = np.random.default_rng(1)
    n, g = 300, 80
    dense = rng.normal(size=(n, g)).astype(np.float32)
    dense[dense < 0.3] = 0.0
    X = sp.csr_matrix(dense)

    class FakeAnnData:
        def __init__(self, X):
            self.X = X
            self.obsm = {}
            self.varm = {}
            self.uns = {}
            self.n_vars = X.shape[1]
            self.var = type("V", (), {"columns": []})()

    ad = FakeAnnData(X)
    out = pp.pca(ad, n_comps=8, n_power_iter=6, random_state=123)

    svd = TruncatedSVD(n_components=8, algorithm="randomized", n_iter=6, random_state=123)
    svd.fit(X.astype(np.float32))

    rel = np.abs(out["singular_values"] - svd.singular_values_) / svd.singular_values_
    assert rel.max() < 0.02, f"max relative SV diff = {rel.max():.3f}"


def test_pca_dense_basic():
    from scatlas import pp

    rng = np.random.default_rng(2)
    n, g = 200, 25
    X = rng.normal(size=(n, g)).astype(np.float32)

    class FakeAnnData:
        def __init__(self, X):
            self.X = X
            self.obsm = {}
            self.varm = {}
            self.uns = {}
            self.n_vars = X.shape[1]
            self.var = type("V", (), {"columns": []})()

    ad = FakeAnnData(X)
    out = pp.pca(ad, n_comps=5, random_state=0)
    assert out["embedding"].shape == (n, 5)
    assert out["components"].shape == (5, g)
    # Singular values decreasing
    sv = out["singular_values"]
    for i in range(len(sv) - 1):
        assert sv[i] >= sv[i + 1] - 1e-4


def test_pca_auto_mode_selects_reasonable_rank():
    """n_comps='auto' should recover ~ true rank via Gavish-Donoho
    for a low-rank + noise matrix."""
    from scatlas import pp

    rng = np.random.default_rng(123)
    n, g, rank_true = 1000, 400, 12
    U = rng.normal(size=(n, rank_true)).astype(np.float32)
    V = rng.normal(size=(rank_true, g)).astype(np.float32)
    X = U @ V + 0.08 * rng.normal(size=(n, g)).astype(np.float32)
    X[np.abs(X) < 0.2] = 0
    X = sp.csr_matrix(X.astype(np.float32))

    class FakeAnnData:
        def __init__(self, X):
            self.X = X
            self.obsm = {}
            self.varm = {}
            self.uns = {}
            self.n_vars = X.shape[1]
            self.var = type("V", (), {"columns": []})()

    ad = FakeAnnData(X)
    out = pp.pca(ad, n_comps="auto", random_state=0)
    # GD should identify close to the true rank
    gd = out["auto"]["n_comps_gavish_donoho"]
    assert abs(gd - rank_true) <= 3, f"GD {gd} far from true rank {rank_true}"
    # Final suggested_n_comps is clamped to [15, 50] with +5 margin
    assert 15 <= out["n_comps"] <= 50
    # Embedding matches effective n_comps
    assert out["embedding"].shape == (n, out["n_comps"])
    # uns records the diagnostics
    assert "auto" in ad.uns["pca"]
    assert ad.uns["pca"]["params"]["n_comps_requested"] == "auto"


def test_suggest_n_comps_api():
    """Standalone suggest_n_comps returns expected dict shape."""
    from scatlas.pp import suggest_n_comps

    svs = np.array(
        [100, 95, 90, 85, 80, 30, 5, 4, 3, 2, 1.5, 1.0] + [0.9] * 20,
        dtype=np.float32,
    )
    res = suggest_n_comps(svs, n_rows=500, n_cols=500)
    assert "n_comps_gavish_donoho" in res
    assert "n_comps_elbow" in res
    assert "suggested_n_comps" in res
    # 5 clearly-large SVs should be caught by GD
    assert res["n_comps_gavish_donoho"] >= 5


def test_pca_rejects_zero_center_true():
    from scatlas import pp

    class FakeAnnData:
        def __init__(self, X):
            self.X = X
            self.obsm = {}
            self.varm = {}
            self.uns = {}
            self.n_vars = X.shape[1]
            self.var = type("V", (), {"columns": []})()

    X = sp.csr_matrix(np.random.default_rng(0).normal(size=(50, 20)).astype(np.float32))
    with pytest.raises(NotImplementedError, match="zero_center"):
        pp.pca(FakeAnnData(X), n_comps=3, zero_center=True)

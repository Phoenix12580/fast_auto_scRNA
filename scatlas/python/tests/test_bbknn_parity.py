"""Parity tests for scatlas.ext.bbknn kernel vs a naive Python reference."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scatlas import ext


def _bbknn_reference(
    pca: np.ndarray, batch: np.ndarray, k_per_batch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy brute-force reference. Returns (indices, distances, batches).

    For ties, breaks by smaller cell index ascending (stable argsort) to
    match Rust's sort_unstable_by fallback behavior in practice. Tests
    that might trip ties should use floats with distinct distances.
    """
    n_cells = pca.shape[0]
    batches = np.unique(batch)
    k_total = k_per_batch * len(batches)
    idx = np.full((n_cells, k_total), np.iinfo(np.uint32).max, dtype=np.uint32)
    dst = np.full((n_cells, k_total), np.inf, dtype=np.float32)
    for i in range(n_cells):
        for bi, bv in enumerate(batches):
            cells = np.where(batch == bv)[0]
            d = np.linalg.norm(pca[cells] - pca[i], axis=1)
            order = np.argsort(d, kind="stable")
            k_eff = min(k_per_batch, len(cells))
            for j in range(k_eff):
                idx[i, bi * k_per_batch + j] = cells[order[j]]
                dst[i, bi * k_per_batch + j] = d[order[j]]
    return idx, dst, batches.astype(np.int32)


def test_bbknn_parity_synthetic_2d():
    rng = np.random.default_rng(42)
    n_cells = 100
    pca = rng.normal(size=(n_cells, 5)).astype(np.float32)
    batch = rng.integers(0, 3, size=n_cells).astype(np.int32)

    out = ext.bbknn_kneighbors(pca, batch, neighbors_within_batch=4)
    ref_idx, ref_dst, ref_bat = _bbknn_reference(pca, batch, k_per_batch=4)

    np.testing.assert_array_equal(out["batches"], ref_bat)
    np.testing.assert_array_equal(out["indices"], ref_idx)
    np.testing.assert_allclose(out["distances"], ref_dst, rtol=1e-5, atol=1e-6)


def test_bbknn_parity_30d_pca_3_batches():
    """Realistic PCA-shaped input: 500 cells, 30 PCs, 3 batches."""
    rng = np.random.default_rng(0)
    n_cells = 500
    pca = rng.normal(size=(n_cells, 30)).astype(np.float32)
    batch = rng.integers(0, 3, size=n_cells).astype(np.int32)
    out = ext.bbknn_kneighbors(pca, batch, neighbors_within_batch=3)
    ref_idx, ref_dst, _ = _bbknn_reference(pca, batch, 3)
    np.testing.assert_array_equal(out["indices"], ref_idx)
    np.testing.assert_allclose(out["distances"], ref_dst, rtol=1e-5, atol=1e-6)


def test_bbknn_string_batch_labels_encoded():
    rng = np.random.default_rng(1)
    n_cells = 80
    pca = rng.normal(size=(n_cells, 10)).astype(np.float32)
    batch_int = rng.integers(0, 2, size=n_cells).astype(np.int32)
    batch_str = np.where(batch_int == 0, "A", "B")

    out_int = ext.bbknn_kneighbors(pca, batch_int, neighbors_within_batch=3)
    out_str = ext.bbknn_kneighbors(pca, batch_str, neighbors_within_batch=3)

    # Same neighbor sets (indices/distances) regardless of label encoding
    np.testing.assert_array_equal(out_int["indices"], out_str["indices"])
    np.testing.assert_array_equal(out_int["distances"], out_str["distances"])
    assert out_str["batch_code_map"] == {0: "A", 1: "B"}


def test_bbknn_small_batch_padded():
    # Batch with only 1 cell; k=2 → second slot is u32::MAX / inf
    pca = np.array([[0.0], [1.0], [10.0]], dtype=np.float32)
    batch = np.array([0, 0, 1], dtype=np.int32)
    out = ext.bbknn_kneighbors(pca, batch, neighbors_within_batch=2)
    # Row 2 batch-1 slots (cols 2,3): [2, MAX] / [0, inf]
    assert out["indices"][2, 2] == 2
    assert out["indices"][2, 3] == np.iinfo(np.uint32).max
    assert out["distances"][2, 2] < 1e-6
    assert np.isinf(out["distances"][2, 3])


def test_bbknn_hnsw_recall_vs_brute():
    """HNSW neighbor recall vs brute should be >= 0.95 at ef=32, k=3, 30-D."""
    rng = np.random.default_rng(11)
    n_cells = 2000
    pca = rng.normal(size=(n_cells, 30)).astype(np.float32)
    batch = rng.integers(0, 3, size=n_cells).astype(np.int32)

    brute = ext.bbknn_kneighbors(
        pca, batch, neighbors_within_batch=3, backend="brute"
    )
    hnsw = ext.bbknn_kneighbors(
        pca, batch, neighbors_within_batch=3, backend="hnsw", ef_search=32
    )
    assert brute["backend_used"] == "brute"
    assert hnsw["backend_used"] == "hnsw"

    # Per-cell set overlap
    MAX = np.iinfo(np.uint32).max
    recalls = []
    for i in range(n_cells):
        truth = set(brute["indices"][i][brute["indices"][i] != MAX].tolist())
        got = set(hnsw["indices"][i][hnsw["indices"][i] != MAX].tolist())
        if truth:
            recalls.append(len(truth & got) / len(truth))
    recall = float(np.mean(recalls))
    assert recall >= 0.95, f"hnsw recall {recall:.3f} < 0.95"


def test_bbknn_auto_backend_picks_brute_for_small():
    rng = np.random.default_rng(2)
    pca = rng.normal(size=(200, 20)).astype(np.float32)
    batch = rng.integers(0, 2, size=200).astype(np.int32)
    out = ext.bbknn_kneighbors(pca, batch, neighbors_within_batch=3, backend="auto")
    assert out["backend_used"] == "brute"


def test_bbknn_auto_backend_picks_hnsw_for_large():
    rng = np.random.default_rng(3)
    n = 12000  # > default auto_threshold 5000
    pca = rng.normal(size=(n, 15)).astype(np.float32)
    batch = rng.integers(0, 2, size=n).astype(np.int32)
    out = ext.bbknn_kneighbors(
        pca, batch, neighbors_within_batch=3, backend="auto"
    )
    assert out["backend_used"] == "hnsw"


def test_bbknn_adata_wrapper():
    """End-to-end scanpy-compat: writes obsp + uns in adata."""
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(7)
    n_cells, n_genes = 120, 50
    X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    adata = anndata.AnnData(X=X)
    adata.obsm["X_pca"] = rng.normal(size=(n_cells, 15)).astype(np.float32)
    adata.obs["batch"] = np.where(rng.integers(0, 2, size=n_cells) == 0, "L", "R")

    result = ext.bbknn(
        adata,
        batch_key="batch",
        use_rep="X_pca",
        neighbors_within_batch=3,
        with_connectivities=False,  # avoid umap-learn requirement
    )
    assert "bbknn_distances" in adata.obsp
    assert "bbknn" in adata.uns
    dist = adata.obsp["bbknn_distances"]
    assert sp.issparse(dist)
    assert dist.shape == (n_cells, n_cells)
    # No padding should reach the sparse matrix
    assert (dist.data >= 0).all()
    assert not np.isinf(dist.data).any()

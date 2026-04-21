"""Parity tests for scatlas.metrics kernels.

LISI: compare to an in-file numpy reference (same Gaussian+perplexity
formulation as scib_metrics / harmonypy). Tolerance is loose on the
perplexity calibration (f32 + bisection convergence).

graph_connectivity: compare to scipy.sparse.csgraph connected_components
applied per label.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from scatlas import metrics


# -----------------------------------------------------------------------------
# LISI reference (numpy, f32)
# -----------------------------------------------------------------------------


def _find_sigma(d2: np.ndarray, log_perp_target: float) -> float:
    beta = 1.0
    lo, hi = -np.inf, np.inf
    for _ in range(50):
        w = np.exp(-d2 * beta - (-d2 * beta).max())
        w /= w.sum()
        nz = w[w > 0]
        h = float(-(nz * np.log2(nz)).sum())
        diff = h - log_perp_target
        if abs(diff) < 1e-5:
            break
        if diff > 0:
            lo = beta
            beta = beta * 2 if not np.isfinite(hi) else (beta + hi) * 0.5
        else:
            hi = beta
            beta = beta * 0.5 if not np.isfinite(lo) else (beta + lo) * 0.5
        if not np.isfinite(beta) or beta <= 0:
            break
    return float(beta)


def _lisi_reference(
    knn_dists: np.ndarray, knn_labels: np.ndarray, perplexity: float
) -> np.ndarray:
    n_cells, k = knn_dists.shape
    log_perp = float(np.log2(perplexity))
    out = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        d = knn_dists[i].astype(np.float32)
        l = knn_labels[i]
        mask = l != np.iinfo(np.int32).min
        d = d[mask]
        l = l[mask]
        if len(d) <= 1:
            out[i] = 1.0
            continue
        d2 = d * d
        beta = _find_sigma(d2, log_perp)
        w = np.exp(-d2 * beta - (-d2 * beta).max())
        w /= w.sum()
        uniq = np.unique(l)
        simpson = 0.0
        for u in uniq:
            simpson += w[l == u].sum() ** 2
        out[i] = 1.0 / simpson if simpson > 0 else 1.0
    return out


def test_lisi_matches_numpy_reference():
    rng = np.random.default_rng(0)
    n_cells, k = 500, 15
    knn_dists = rng.exponential(scale=1.0, size=(n_cells, k)).astype(np.float32)
    knn_dists.sort(axis=1)  # sorted ascending — typical k-NN layout
    knn_labels = rng.integers(0, 3, size=(n_cells, k)).astype(np.int32)

    ours = metrics.lisi(knn_dists, knn_labels, perplexity=5.0)
    ref = _lisi_reference(knn_dists, knn_labels, perplexity=5.0)
    # f32 bisection → ~1e-4 tolerance
    np.testing.assert_allclose(ours, ref, rtol=1e-3, atol=1e-3)


def test_lisi_uniform_two_label():
    d = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    l = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int32)
    out = metrics.lisi(d, l, perplexity=4.0)
    assert abs(out[0] - 2.0) < 1e-3


def test_lisi_string_labels_encoded():
    d = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    # Non-integer labels
    l = np.array([["a", "a", "b", "b"]])
    out = metrics.lisi(d, l, perplexity=3.0)
    assert 1.5 < out[0] <= 2.0, out


def test_ilisi_clisi_range_and_direction():
    # Scenario: 2 batches perfectly mixed, 2 cell types perfectly preserved
    rng = np.random.default_rng(42)
    n_cells, k = 300, 20
    # Make neighbors draw from both batches uniformly, matching labels
    # only per block: cells 0..149 are label A, cells 150..299 label B
    labels = np.concatenate([np.zeros(150), np.ones(150)]).astype(np.int32)
    batches = rng.integers(0, 2, size=n_cells).astype(np.int32)

    # Build perfect-mixing kNN: neighbors = random cells from same label
    # but balanced across batches.
    knn_idx = np.zeros((n_cells, k), dtype=np.int64)
    knn_dists = np.ones((n_cells, k), dtype=np.float32)
    for i in range(n_cells):
        same_label = np.where(labels == labels[i])[0]
        choice = rng.choice(same_label, size=k, replace=False)
        knn_idx[i] = choice

    # Expand per-neighbor batch and label codes
    nbr_batch = batches[knn_idx].astype(np.int32)
    nbr_label = labels[knn_idx].astype(np.int32)

    ilisi = metrics.ilisi(knn_dists, nbr_batch, perplexity=10.0)
    clisi = metrics.clisi(knn_dists, nbr_label, perplexity=10.0)
    assert 0.5 <= ilisi <= 1.0, f"ilisi={ilisi} should be high (mixed batches)"
    assert 0.8 <= clisi <= 1.0, f"clisi={clisi} should be high (preserved labels)"


# -----------------------------------------------------------------------------
# graph_connectivity
# -----------------------------------------------------------------------------


def _graph_connectivity_reference(
    knn_indices: np.ndarray, labels: np.ndarray
) -> float:
    """Reference via scipy.sparse.csgraph per-label connected_components."""
    n_cells, k = knn_indices.shape
    MAX = np.iinfo(np.uint32).max
    fracs = []
    for lbl in np.unique(labels):
        cells = np.where(labels == lbl)[0]
        if len(cells) == 0:
            continue
        if len(cells) == 1:
            fracs.append(1.0)
            continue
        cell_set = set(cells.tolist())
        rows = []
        cols = []
        for c in cells:
            for j in range(k):
                v = int(knn_indices[c, j])
                if v == MAX:
                    continue
                if v in cell_set:
                    rows.append(c)
                    cols.append(v)
        if len(rows) == 0:
            fracs.append(1.0 / len(cells))
            continue
        sub = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_cells, n_cells),
        )
        # Restrict to rows/cols in this label
        sub_sub = sub[cells][:, cells]
        n_cc, lab_cc = connected_components(sub_sub, directed=False)
        _, counts = np.unique(lab_cc, return_counts=True)
        fracs.append(counts.max() / len(cells))
    return float(np.mean(fracs))


def test_graph_connectivity_parity():
    rng = np.random.default_rng(5)
    n_cells, k = 200, 5
    knn_indices = rng.integers(0, n_cells, size=(n_cells, k)).astype(np.uint32)
    # Introduce some padding to stress the implementation
    pad_mask = rng.random((n_cells, k)) < 0.1
    knn_indices[pad_mask] = np.iinfo(np.uint32).max
    labels = rng.integers(0, 4, size=n_cells).astype(np.int32)

    ours = metrics.graph_connectivity(knn_indices, labels)
    ref = _graph_connectivity_reference(knn_indices, labels)
    assert abs(ours - ref) < 1e-9, f"ours={ours}  ref={ref}"


def test_graph_connectivity_perfect():
    # 10 cells, 2 labels of 5, each block fully internally connected.
    knn = np.zeros((10, 4), dtype=np.uint32)
    for i in range(5):
        # Label 0 (cells 0-4): each points to other 4
        neighbors = [j for j in range(5) if j != i]
        knn[i] = neighbors
    for i in range(5, 10):
        neighbors = [j for j in range(5, 10) if j != i]
        knn[i] = neighbors
    labels = np.array([0] * 5 + [1] * 5, dtype=np.int32)
    r = metrics.graph_connectivity(knn, labels)
    assert abs(r - 1.0) < 1e-12


def test_graph_connectivity_accepts_sparse_input():
    # scanpy-style connectivities csr with shuffled indices
    rng = np.random.default_rng(11)
    n = 50
    row, col = [], []
    for i in range(n):
        for _ in range(3):
            j = rng.integers(0, n)
            if j != i:
                row.append(i)
                col.append(j)
    sp_mat = sp.csr_matrix(
        (np.ones(len(row)), (row, col)), shape=(n, n)
    )
    labels = rng.integers(0, 3, size=n).astype(np.int32)
    r = metrics.graph_connectivity(sp_mat, labels)
    assert 0.0 <= r <= 1.0


def test_scib_score_composes():
    """scib_score returns expected keys, values in [0, 1]."""
    rng = np.random.default_rng(99)
    n_cells, k = 400, 15
    knn_idx = rng.integers(0, n_cells, size=(n_cells, k)).astype(np.uint32)
    knn_dists = rng.exponential(size=(n_cells, k)).astype(np.float32)
    batch = rng.integers(0, 2, size=n_cells).astype(np.int32)
    cell_type = rng.integers(0, 5, size=n_cells).astype(np.int32)

    out = metrics.scib_score(knn_idx, knn_dists, batch, cell_type, perplexity=10.0)
    expected_keys = {
        "ilisi", "clisi", "graph_connectivity", "kbet_acceptance", "mean",
    }
    assert set(out) == expected_keys, out.keys()
    for key in expected_keys:
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]}"


def test_scib_score_with_embedding_adds_silhouettes():
    rng = np.random.default_rng(11)
    n_cells, k = 200, 10
    knn_idx = rng.integers(0, n_cells, size=(n_cells, k)).astype(np.uint32)
    knn_dists = rng.exponential(size=(n_cells, k)).astype(np.float32)
    batch = rng.integers(0, 2, size=n_cells).astype(np.int32)
    cell_type = rng.integers(0, 3, size=n_cells).astype(np.int32)
    pca = rng.normal(size=(n_cells, 8)).astype(np.float32)

    out = metrics.scib_score(
        knn_idx, knn_dists, batch, cell_type, perplexity=5.0, embedding=pca
    )
    for key in ("label_silhouette", "batch_silhouette"):
        assert key in out
        assert 0.0 <= out[key] <= 1.0


# -----------------------------------------------------------------------------
# kBET
# -----------------------------------------------------------------------------


def _kbet_reference(
    knn_indices: np.ndarray, batch_labels: np.ndarray, alpha: float
) -> float:
    """scipy-based reference: same χ² formula, compared against p ≥ alpha."""
    from scipy.stats import chi2 as chi2_dist

    _, codes = np.unique(batch_labels, return_inverse=True)
    n_batches = len(np.unique(codes))
    _, counts = np.unique(codes, return_counts=True)
    n_total = counts.sum()
    MAX = np.iinfo(np.uint32).max

    n_cells, k = knn_indices.shape
    accept = 0
    for i in range(n_cells):
        obs = np.zeros(n_batches, dtype=np.float64)
        k_eff = 0
        for j in range(k):
            v = int(knn_indices[i, j])
            if v == MAX:
                continue
            obs[codes[v]] += 1
            k_eff += 1
        if k_eff == 0:
            accept += 1
            continue
        exp = k_eff * counts / n_total
        chi2 = float(((obs - exp) ** 2 / exp).sum())
        p = chi2_dist.sf(chi2, n_batches - 1)
        if p >= alpha:
            accept += 1
    return accept / n_cells


def test_kbet_matches_scipy_reference():
    rng = np.random.default_rng(7)
    n_cells, k = 300, 10
    knn_idx = rng.integers(0, n_cells, size=(n_cells, k)).astype(np.uint32)
    # Add some padding
    pad = rng.random((n_cells, k)) < 0.08
    knn_idx[pad] = np.iinfo(np.uint32).max
    batch = rng.integers(0, 3, size=n_cells)  # 3 batches

    ours = metrics.kbet(knn_idx, batch, alpha=0.05)
    ref = _kbet_reference(knn_idx, batch, alpha=0.05)
    assert abs(ours["acceptance_rate"] - ref) < 1e-10


def test_kbet_perfect_integration_rate_one():
    """Every cell's neighbors mirror the global 50/50 → χ²=0 → p=1 → accepted."""
    n_cells = 100
    batch = np.array([0, 1] * (n_cells // 2), dtype=np.int32)
    # knn_indices: each cell's neighbors alternate between the two batches
    # so 3 from each batch (k=6).
    k = 6
    knn_idx = np.zeros((n_cells, k), dtype=np.uint32)
    b0 = np.where(batch == 0)[0]
    b1 = np.where(batch == 1)[0]
    for i in range(n_cells):
        knn_idx[i, :3] = b0[:3]
        knn_idx[i, 3:] = b1[:3]
    out = metrics.kbet(knn_idx, batch, alpha=0.05)
    assert out["acceptance_rate"] > 0.99


def test_kbet_single_batch_returns_one():
    batch = np.zeros(20, dtype=np.int32)
    knn_idx = np.zeros((20, 3), dtype=np.uint32)
    out = metrics.kbet(knn_idx, batch)
    assert out["acceptance_rate"] == 1.0
    assert out["n_batches"] == 1


def test_label_batch_silhouette_work():
    rng = np.random.default_rng(3)
    n = 120
    pca = rng.normal(size=(n, 6)).astype(np.float32)
    cell_type = rng.integers(0, 3, size=n)
    batch = rng.integers(0, 2, size=n)

    lab = metrics.label_silhouette(pca, cell_type)
    bat = metrics.batch_silhouette(pca, batch, cell_type)
    assert 0.0 <= lab <= 1.0
    assert 0.0 <= bat <= 1.0

"""Rust silhouette_precomputed kernel vs sklearn consistency test.

Guards the GS-3 milestone: the Rust kernel that replaces the sklearn
fallback in ``cluster/resolution.py::_silhouette_impl`` must produce
numerically equivalent scores (same mean silhouette to within a tiny
float tolerance) on synthetic data with known cluster structure.

If this test starts failing the graph-silhouette resolution selector
will pick different resolutions than v1's sklearn sweep did, silently
changing 222k atlas cluster counts — a regression we cannot ship.
"""
from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("fast_auto_scrna._native", reason="run `maturin develop` first")
pytest.importorskip(
    "fast_auto_scrna._native.silhouette",
    reason="GS-3 Rust silhouette kernel not yet built (maturin develop needed)",
)


def _synthetic_distance_matrix(
    n: int, k: int, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n, n) distance + (n,) labels with k well-separated clusters.

    Within-cluster distances ~ 0.1; between-cluster ~ 1-3 depending on
    cluster centroids. Returns (distances float32, labels int32).
    """
    rng = np.random.default_rng(seed)
    # Place cluster centroids on a k-simplex in 5D
    centroids = rng.standard_normal((k, 5)).astype(np.float32) * 3.0
    # Assign points to clusters (balanced-ish)
    labels = rng.integers(0, k, size=n).astype(np.int32)
    # Generate point coordinates around centroids
    coords = centroids[labels] + rng.standard_normal((n, 5)).astype(np.float32) * 0.3
    # Pairwise Euclidean distances
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)
    return d, labels


@pytest.mark.parametrize(
    "n,k,seed",
    [
        (200, 3, 0),
        (500, 5, 1),
        (1000, 7, 2),
    ],
)
def test_rust_vs_sklearn_silhouette(n: int, k: int, seed: int):
    from sklearn.metrics import silhouette_score
    from fast_auto_scrna._native import silhouette as _native_sil

    d, labels = _synthetic_distance_matrix(n, k, seed=seed)

    # sklearn reference
    ref = silhouette_score(d, labels, metric="precomputed")

    # Rust kernel
    got = _native_sil.silhouette_precomputed(d, labels)

    # Rust uses f32 throughout, sklearn is f64 internally. Tolerate small
    # float difference; 1e-4 is extremely tight for f32.
    assert abs(ref - got) < 1e-4, (
        f"silhouette mismatch: sklearn={ref:.6f} vs Rust={got:.6f} "
        f"(n={n}, k={k}, seed={seed})"
    )


def test_rust_silhouette_singleton_cluster():
    """Rust kernel must treat singleton-cluster members as s=0, matching
    our Python fallback. sklearn's own behavior is to raise — we don't."""
    from fast_auto_scrna._native import silhouette as _native_sil

    # 3 points: cluster 0 has 2 members, cluster 1 is a singleton
    d = np.array([
        [0.0, 0.1, 5.0],
        [0.1, 0.0, 5.0],
        [5.0, 5.0, 0.0],
    ], dtype=np.float32)
    l = np.array([0, 0, 1], dtype=np.int32)
    s = _native_sil.silhouette_precomputed(d, l)
    # i=0: a=0.1, b=5.0 → s=0.98
    # i=1: a=0.1, b=5.0 → s=0.98
    # i=2: singleton → 0
    # mean = 0.653
    assert abs(s - 0.653) < 0.01, f"expected ~0.653, got {s}"


def test_rust_silhouette_single_cluster_returns_zero():
    """Single-cluster input → 0 (our convention; sklearn would raise)."""
    from fast_auto_scrna._native import silhouette as _native_sil

    d = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    l = np.array([0, 0], dtype=np.int32)
    s = _native_sil.silhouette_precomputed(d, l)
    assert s == 0.0

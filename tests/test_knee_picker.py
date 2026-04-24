"""Knee-picker tests.

1. perpendicular_elbow matches the Rust kernel's PCA scree convention on
   a known descending exponential.
2. perpendicular_elbow on a synthetic two-step ascending conductance-like
   curve hits the step.
3. End-to-end optimize_resolution_knee on a 2-blob synthetic graph picks a
   resolution giving k ≥ 2 (not the trivial k=1).
"""
from __future__ import annotations

import numpy as np
import pytest


def test_perpendicular_elbow_descending_exponential():
    """Classic scree: strong early drop then slow tail. Knee should sit
    near the bend (around idx 3-5 for decay-rate 0.6)."""
    from fast_auto_scrna.cluster.resolution import perpendicular_elbow
    y = np.exp(-0.6 * np.arange(20, dtype=np.float64))
    knee = perpendicular_elbow(y)
    assert 2 <= knee <= 7, f"expected knee near the bend, got idx={knee}"


def test_perpendicular_elbow_ascending_step():
    """Two-plateau ascending function: low plateau idx 0-4, jump at 5,
    high plateau idx 5-9. Knee should land at idx 4 (last low point) or
    idx 5 (first high point)."""
    from fast_auto_scrna.cluster.resolution import perpendicular_elbow
    y = np.array([0.01, 0.01, 0.01, 0.01, 0.01,
                  0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float64)
    knee = perpendicular_elbow(y)
    assert knee in (4, 5), f"expected knee at step boundary, got idx={knee}"


def test_first_plateau_after_rise_prefers_first_step():
    """Multi-step ascending: low plateau, then small first rise to
    mid-plateau, then big second rise. first_plateau_after_rise should
    find the FIRST plateau entry (after the first rise), not the second
    one — unlike perpendicular_elbow which is pulled by the global slope.
    """
    from fast_auto_scrna.cluster.resolution import first_plateau_after_rise
    # low plateau (idx 0-4) → rise → mid plateau (idx 9-13) → bigger rise
    # → high plateau (idx 18-22)
    y = np.array(
        [0.01]*5 + [0.05, 0.10, 0.13, 0.14] + [0.15]*5 +
        [0.30, 0.50, 0.65, 0.70] + [0.72]*5,
        dtype=np.float64,
    )
    knee = first_plateau_after_rise(
        y, window=5, min_rise_ratio=0.10, low_slope_ratio=0.25,
    )
    # The first plateau entry is around idx 9 (end of the first rise)
    # not the second plateau around idx 18.
    assert 8 <= knee <= 12, f"expected first-plateau entry near idx 9, got {knee}"


def test_optimize_resolution_knee_two_blob():
    """Synthetic 2-blob graph — knee picker must not return trivial k=1."""
    import anndata as ad
    import scipy.sparse as sp
    import scanpy as sc
    from fast_auto_scrna.cluster.resolution import optimize_resolution_knee

    rng = np.random.default_rng(42)
    n = 400
    X1 = rng.standard_normal((n // 2, 10)).astype(np.float32)
    X2 = rng.standard_normal((n // 2, 10)).astype(np.float32) + 6.0
    X = np.vstack([X1, X2])
    a = ad.AnnData(X=X)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X")
    G = a.obsp["connectivities"].tocsr()
    a.obsp["bbknn_connectivities"] = G

    resolutions = [round(r, 2) for r in np.arange(0.05, 1.55, 0.05)]
    curve = optimize_resolution_knee(
        a, method="bbknn",
        resolutions=resolutions,
        offset_steps=3, seed=0, leiden_n_iterations=2, verbose=False,
    )
    picked_row = curve.loc[curve["is_picked"]].iloc[0]
    assert int(picked_row["n_clusters"]) >= 2, (
        f"knee picker returned trivial k={int(picked_row['n_clusters'])}, "
        f"curve: {curve}"
    )

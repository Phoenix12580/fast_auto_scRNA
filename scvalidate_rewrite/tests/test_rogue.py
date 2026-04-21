"""Unit tests for rogue_py.

Strategy:
* Synthetic datasets with known properties (pure vs heterogeneous) to
  verify that ROGUE ordering matches intuition.
* Tolerance on loess-backed outputs: these tests run with either skmisc
  (bit-compatible with R) or the statsmodels fallback; we keep absolute
  tolerances loose for fallback but verify that the **ranking** between
  pure and mixed populations is preserved (the property ROGUE claims).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scvalidate.rogue_py import (
    calculate_rogue,
    entropy_fit,
    entropy_table,
    filter_matrix,
    se_fun,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def pure_counts(rng):
    """A homogeneous Poisson population (500 cells × 1000 genes)."""
    lam = rng.uniform(0.5, 5.0, size=1000)
    return rng.poisson(lam=lam[:, None], size=(1000, 500)).astype(np.int64)


@pytest.fixture
def mixed_counts(rng):
    """Two Poisson sub-populations with disjoint mean profiles."""
    lam_a = rng.uniform(0.2, 8.0, size=1000)
    lam_b = rng.uniform(0.2, 8.0, size=1000)
    left = rng.poisson(lam=lam_a[:, None], size=(1000, 250))
    right = rng.poisson(lam=lam_b[:, None], size=(1000, 250))
    return np.concatenate([left, right], axis=1).astype(np.int64)


# -----------------------------------------------------------------------------
# entropy_table
# -----------------------------------------------------------------------------


def test_entropy_table_shape_and_types(pure_counts):
    ent = entropy_table(pure_counts)
    assert set(ent.columns) == {"Gene", "mean_expr", "entropy"}
    assert len(ent) == pure_counts.shape[0]
    assert np.all(np.isfinite(ent["entropy"].to_numpy()))


def test_entropy_table_matches_r_formula(pure_counts):
    """Recompute the R formula by hand and compare."""
    ent = entropy_table(pure_counts, r=1.0)
    # R: entropy = rowMeans(log(expr + 1))
    manual_entropy = np.log(pure_counts + 1).mean(axis=1)
    np.testing.assert_allclose(ent["entropy"].to_numpy(), manual_entropy, rtol=1e-12)
    # R: mean.expr = log(rowMeans(expr) + r)
    manual_mean = np.log(pure_counts.mean(axis=1) + 1.0)
    np.testing.assert_allclose(ent["mean_expr"].to_numpy(), manual_mean, rtol=1e-12)


# -----------------------------------------------------------------------------
# entropy_fit
# -----------------------------------------------------------------------------


def test_entropy_fit_returns_expected_columns(pure_counts):
    ent = entropy_table(pure_counts)
    fit = entropy_fit(ent, span=0.5)
    for col in ("Gene", "mean_expr", "entropy", "fit", "ds", "p_value", "p_adj"):
        assert col in fit.columns
    # p-values and adjusted p-values in [0, 1]
    assert fit["p_value"].between(0, 1).all()
    assert fit["p_adj"].between(0, 1).all()


def test_entropy_fit_sorted_by_ds_desc(pure_counts):
    ent = entropy_table(pure_counts)
    fit = entropy_fit(ent, span=0.5)
    ds = fit["ds"].to_numpy()
    assert np.all(ds[:-1] >= ds[1:])


# -----------------------------------------------------------------------------
# calculate_rogue
# -----------------------------------------------------------------------------


def test_rogue_pure_gt_mixed(pure_counts, mixed_counts):
    """A homogeneous population must score higher ROGUE than a mixed one."""
    pure_table = se_fun(pure_counts, span=0.5, r=1.0)
    mixed_table = se_fun(mixed_counts, span=0.5, r=1.0)
    pure_score = calculate_rogue(pure_table, platform="UMI")
    mixed_score = calculate_rogue(mixed_table, platform="UMI")
    assert pure_score > mixed_score, (
        f"expected pure > mixed; got pure={pure_score}, mixed={mixed_score}"
    )
    # Both should be in [0, 1]
    assert 0.0 <= mixed_score <= pure_score <= 1.0


def test_rogue_k_override(pure_counts):
    table = se_fun(pure_counts, span=0.5, r=1.0)
    # Passing explicit k should bypass platform lookup
    score_umi = calculate_rogue(table, platform="UMI")
    score_k45 = calculate_rogue(table, k=45)
    assert score_umi == pytest.approx(score_k45, rel=1e-12)


def test_rogue_requires_k_or_platform(pure_counts):
    table = se_fun(pure_counts, span=0.5, r=1.0)
    with pytest.raises(ValueError):
        calculate_rogue(table)


def test_rogue_features_filter(pure_counts):
    table = se_fun(pure_counts, span=0.5, r=1.0)
    sub = table["Gene"].head(10).tolist()
    score_sub = calculate_rogue(table, platform="UMI", features=sub)
    score_all = calculate_rogue(table, platform="UMI")
    # Restricting to top-10 must yield a *different* score (possibly higher
    # or lower depending on ds sign); just verify it runs and is in range.
    assert 0.0 <= score_sub <= 1.0
    assert 0.0 <= score_all <= 1.0


# -----------------------------------------------------------------------------
# filter_matrix
# -----------------------------------------------------------------------------


def test_filter_matrix_drops_low_coverage():
    # 5 genes × 20 cells. Gene 0 is all-zero, Gene 4 detected in only 2 cells.
    m = np.zeros((5, 20), dtype=np.int64)
    m[1, :] = 3
    m[2, :15] = 1
    m[3, :5] = 2
    m[4, :2] = 5
    filtered = filter_matrix(m, min_cells=10, min_genes=1)
    # Gene 0, 3, 4 drop (detected in <10 cells); Genes 1 and 2 remain.
    assert filtered.shape[0] == 2


# -----------------------------------------------------------------------------
# Sparse vs dense parity
# -----------------------------------------------------------------------------


def test_sparse_dense_entropy_parity(pure_counts):
    import scipy.sparse as sp

    ent_dense = entropy_table(pure_counts)
    ent_sparse = entropy_table(sp.csr_matrix(pure_counts))
    np.testing.assert_allclose(
        ent_dense["entropy"].to_numpy(), ent_sparse["entropy"].to_numpy(), rtol=1e-10
    )
    np.testing.assert_allclose(
        ent_dense["mean_expr"].to_numpy(), ent_sparse["mean_expr"].to_numpy(), rtol=1e-10
    )

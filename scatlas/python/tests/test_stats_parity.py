"""Parity tests: scatlas.stats.* vs scipy/numpy reference implementations.

Rationale: scatlas-core is a port of scvalidate_rust, which itself was
parity-tested against scvalidate.recall_py / scvalidate.rogue_py. Rather
than pull scvalidate (heavy deps: scanpy, pandas, R bridges) into the CI
venv, these tests re-derive the gold standard locally from scipy/numpy
primitives, matching the exact formulas scvalidate uses.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import erfc
from scipy.stats import rankdata

from scatlas import stats


# -----------------------------------------------------------------------------
# Wilcoxon
# -----------------------------------------------------------------------------


def _wilcoxon_reference(
    x: np.ndarray, m1: np.ndarray, m2: np.ndarray
) -> np.ndarray:
    """scvalidate's Mann-Whitney U formula: normal-approx, no ties correction."""
    n_genes = x.shape[0]
    union = m1 | m2
    group1_in_union = m1[union]
    n1 = int(group1_in_union.sum())
    n2 = int((~group1_in_union).sum())
    if n1 == 0 or n2 == 0:
        return np.ones(n_genes)
    mu = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    out = np.zeros(n_genes)
    for g in range(n_genes):
        ranks = rankdata(x[g, union], method="average")
        r1 = ranks[group1_in_union].sum()
        u1 = r1 - n1 * (n1 + 1) / 2.0
        z = (u1 - mu) / sigma
        out[g] = np.clip(erfc(abs(z) / np.sqrt(2)), 1e-300, 1.0)
    return out


def test_wilcoxon_parity_f64():
    rng = np.random.default_rng(42)
    n_genes, n_cells = 30, 200
    x = rng.normal(size=(n_genes, n_cells)).astype(np.float64)
    x[:15, :100] += 1.0  # signal in first half of genes
    m1 = np.array([True] * 100 + [False] * 100)
    m2 = np.array([False] * 100 + [True] * 100)

    ours = stats.wilcoxon_ranksum_matrix(x, m1, m2)
    ref = _wilcoxon_reference(x, m1, m2)
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12)


def test_wilcoxon_ties_handled():
    x = np.array(
        [
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],  # lots of ties
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],  # alternating ties
        ]
    )
    m1 = np.array([True, True, True, False, False, False])
    m2 = np.array([False, False, False, True, True, True])
    ours = stats.wilcoxon_ranksum_matrix(x, m1, m2)
    ref = _wilcoxon_reference(x, m1, m2)
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12)


def test_wilcoxon_f32_matches_f64_within_tolerance():
    rng = np.random.default_rng(7)
    x64 = rng.normal(size=(20, 100)).astype(np.float64)
    x64[:10, :50] += 0.8
    x32 = x64.astype(np.float32)
    m1 = np.array([True] * 50 + [False] * 50)
    m2 = np.array([False] * 50 + [True] * 50)

    p64 = stats.wilcoxon_ranksum_matrix(x64, m1, m2)
    p32 = stats.wilcoxon_ranksum_matrix(x32, m1, m2)
    # f32 ties can break differently near equal values; allow loose tolerance
    np.testing.assert_allclose(p64, p32, rtol=1e-4, atol=1e-4)


def test_wilcoxon_empty_group_returns_ones():
    x = np.array([[1.0, 2.0, 3.0]])
    m1 = np.array([True, True, True])
    m2 = np.array([False, False, False])
    p = stats.wilcoxon_ranksum_matrix(x, m1, m2)
    np.testing.assert_array_equal(p, [1.0])


def test_wilcoxon_shape_mismatch_raises():
    x = np.array([[1.0, 2.0, 3.0]])
    m1 = np.array([True, False])
    m2 = np.array([False, True])
    with pytest.raises(ValueError):
        stats.wilcoxon_ranksum_matrix(x, m1, m2)


# -----------------------------------------------------------------------------
# Knockoff
# -----------------------------------------------------------------------------


def _knockoff_reference(w: np.ndarray, fdr: float) -> float:
    ts = np.sort(np.unique(np.abs(w[w != 0])))
    if len(ts) == 0:
        return float("inf")
    for t in ts:
        neg = int((w <= -t).sum())
        pos = int((w >= t).sum())
        num = 1.0 + neg
        denom = max(1, pos)
        if num / denom <= fdr:
            return float(t)
    return float("inf")


def test_knockoff_parity_vs_numpy():
    rng = np.random.default_rng(0)
    w = rng.normal(size=5000)
    for fdr in (0.05, 0.1, 0.2):
        ours = stats.knockoff_threshold_offset1(w, fdr)
        ref = _knockoff_reference(w, fdr)
        assert ours == ref, f"fdr={fdr} ours={ours} ref={ref}"


def test_knockoff_all_zero_returns_inf():
    assert stats.knockoff_threshold_offset1(np.zeros(10), 0.1) == float("inf")


# -----------------------------------------------------------------------------
# ROGUE
# -----------------------------------------------------------------------------


def _entropy_table_reference(counts: np.ndarray, r: float) -> np.ndarray:
    mean = counts.mean(axis=1)
    mean_expr = np.log(mean + r)
    entropy = np.log(counts + 1.0).mean(axis=1)
    return np.column_stack([mean_expr, entropy])


def test_rogue_entropy_dense_parity():
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=2.0, size=(100, 300)).astype(np.float64)
    ours = stats.entropy_table(counts, r=1.0)
    ref = _entropy_table_reference(counts, r=1.0)
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12)


def test_rogue_entropy_sparse_matches_dense():
    rng = np.random.default_rng(1)
    dense = rng.poisson(lam=0.4, size=(200, 1000)).astype(np.float64)
    sparse = sp.csr_matrix(dense)
    d = stats.entropy_table(dense, r=1.0)
    s = stats.entropy_table(sparse, r=1.0)
    np.testing.assert_allclose(d, s, rtol=1e-10, atol=1e-12)


def test_rogue_entropy_respects_pseudocount_r():
    counts = np.array([[0.0, 10.0], [1.0, 1.0]])
    for r in (0.1, 1.0, 5.0):
        ours = stats.entropy_table(counts, r=r)
        ref = _entropy_table_reference(counts, r=r)
        np.testing.assert_allclose(ours, ref, rtol=1e-12)


def test_calculate_rogue_umi_full_length_defaults():
    ds = np.array([2.0, -3.0])
    p_adj = np.array([0.01, 0.01])
    p_value = np.array([0.01, 0.04])
    umi = stats.calculate_rogue(ds, p_adj, p_value, platform="UMI")
    fl = stats.calculate_rogue(ds, p_adj, p_value, platform="full-length")
    assert abs(umi - (1 - 5 / 50)) < 1e-12
    assert abs(fl - (1 - 5 / 505)) < 1e-12


def test_calculate_rogue_rejects_missing_k_and_platform():
    ds = np.array([2.0])
    p_adj = np.array([0.01])
    p_value = np.array([0.01])
    with pytest.raises(ValueError):
        stats.calculate_rogue(ds, p_adj, p_value)

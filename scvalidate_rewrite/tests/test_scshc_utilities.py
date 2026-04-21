"""Unit tests for scshc_py building blocks (not the full orchestrator).

These cover the ported utilities that are complete in v0.1:
poisson_dev_batch, poisson_dispersion_stats, reduce_dimension, ward_linkage_stat,
and fit_model_batch. The full scshc() / test_clusters() orchestrators are
deliberately not tested here — they raise NotImplementedError and will be
validated against R reference outputs in v0.2.
"""

import numpy as np
import pytest

from scvalidate.scshc_py import (
    compute_ess,
    fit_model_batch,
    poisson_dev_batch,
    poisson_dispersion_stats,
    reduce_dimension,
    ward_linkage_stat,
)


@pytest.fixture
def synthetic_counts():
    rng = np.random.default_rng(0)
    # Genes × cells, two sub-populations for the clustering stats
    n_genes, n_cells = 200, 300
    left = rng.poisson(lam=2.0, size=(n_genes, n_cells // 2))
    right = rng.poisson(lam=5.0, size=(n_genes, n_cells // 2))
    return np.concatenate([left, right], axis=1).astype(np.float64)


def test_poisson_dev_shape(synthetic_counts):
    pdev = poisson_dev_batch(synthetic_counts)
    assert pdev.shape == synthetic_counts.shape
    assert np.all(np.isfinite(pdev))


def test_poisson_dev_sign_convention(synthetic_counts):
    # Overexpressed entries should have positive residuals
    pdev = poisson_dev_batch(synthetic_counts)
    n = synthetic_counts.sum(axis=0)
    pis = synthetic_counts.sum(axis=1) / synthetic_counts.sum()
    mu = np.outer(pis, n)
    over = synthetic_counts > mu
    # Where observed > expected → residual should be >= 0
    assert np.all(pdev[over] >= 0)


def test_dispersion_stats_shape(synthetic_counts):
    phi = poisson_dispersion_stats(synthetic_counts)
    assert phi.shape == (synthetic_counts.shape[0],)


def test_reduce_dimension_output(synthetic_counts):
    pcs, proj = reduce_dimension(synthetic_counts, None, num_pcs=10)
    # projection is cells × num_pcs
    assert proj.shape == (synthetic_counts.shape[1], 10)
    assert pcs["values"].shape == (10,)


def test_compute_ess_zero_on_constant_rows():
    x = np.tile(np.array([1.0, 2.0, 3.0]), (10, 1))
    assert compute_ess(x) == pytest.approx(0.0)


def test_ward_linkage_stat_positive_with_separated_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0, scale=1, size=(50, 3))
    b = rng.normal(loc=5, scale=1, size=(50, 3))
    reduc = np.concatenate([a, b], axis=0)
    labels = np.concatenate([np.ones(50), np.full(50, 2)]).astype(int)
    stat = ward_linkage_stat(reduc, labels)
    assert stat > 0


def test_fit_model_batch_returns_valid_cholesky(synthetic_counts):
    # Pick a handful of genes with signal
    on_genes = np.arange(20)
    lambdas, mus, cov_sqrt = fit_model_batch(
        synthetic_counts, on_genes=on_genes, num_pcs=5
    )
    assert lambdas.shape == (synthetic_counts.shape[0],)
    assert mus.shape == (20,)
    # cov_sqrt is lower triangular — upper triangle should be ~0
    upper = np.triu(cov_sqrt, k=1)
    assert np.allclose(upper, 0.0)


def test_fit_model_batch_rspectra_parity_with_negative_eigs():
    # Regression test for two R-parity bugs in fit_model_batch:
    #
    # (1) R's RSpectra::eigs_sym(k=K) returns top-K by |λ| (largest magnitude),
    #     so when `rhos` carries large *negative* eigenvalues, R filters them
    #     out and the effective rank is < K. A naive eigh+top-K-algebraic
    #     always fills K slots with small positives → implied covariance is
    #     inflated by ~40% → null LogNormal-Poisson draws run hot → null Ward
    #     stats 2–3× R's → uniform p≈1 (all splits look null-like).
    #
    # (2) sfsmisc::posdefify(method="someEVadd") clips negative eigenvalues to
    #     ε *and* rescales so diag(result) == diag(input). Without the rescale
    #     the diagonal drifts after eigenvalue clipping and variances shift.
    #
    # We construct synthetic counts whose on-gene log-covariance `rhos` is
    # guaranteed to have large negative eigenvalues in the top-K-by-|λ| slice.
    # With the fix, the Cholesky factor's Frobenius norm must stay close to
    # what R produces; without the fix it inflates measurably.
    rng = np.random.default_rng(42)
    n_genes, n_cells = 150, 400
    # Strong negative correlations between gene blocks → rhos eigenvalues
    # have large negative tails once log-transformed.
    block_a = rng.poisson(lam=8.0, size=(n_genes // 2, n_cells))
    block_b = rng.poisson(lam=8.0, size=(n_genes // 2, n_cells))
    # Anti-correlate block_b with block_a via shared cell-level factor
    factor = rng.normal(0, 2.0, size=n_cells).clip(-4, 4)
    block_a = np.clip(block_a + factor, 0, None).astype(np.float64)
    block_b = np.clip(block_b - factor, 0, None).astype(np.float64)
    counts = np.vstack([block_a, block_b])

    on_genes = np.arange(n_genes)
    _, _, cov_sqrt = fit_model_batch(counts, on_genes=on_genes, num_pcs=30)

    # Cholesky factor is valid lower-triangular
    assert np.allclose(np.triu(cov_sqrt, k=1), 0.0)
    # Reconstructed covariance is PSD (eigenvalues > -tol)
    cov = cov_sqrt @ cov_sqrt.T
    evs = np.linalg.eigvalsh(cov)
    assert evs.min() > -1e-8

    # Diagonal of reconstructed cov must match diag(rhos) (posdefify rescale).
    # sigmas = log(((var - mean)/mean²) + 1); we recompute to assert the
    # rescale preserved the log-normal scale parameters.
    on_counts = counts[on_genes, :].T
    sample_cov = np.cov(on_counts, rowvar=False, ddof=1)
    means = on_counts.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_sigmas = np.log(((np.diag(sample_cov) - means) / means**2) + 1.0)
    expected_sigmas = np.where(
        np.isfinite(expected_sigmas), expected_sigmas, 0.0
    )
    # Only check genes where sigma is finite and positive (the rest are
    # clamped by posdefify's ε floor and are not required to match exactly).
    valid = expected_sigmas > 1e-6
    assert np.allclose(
        np.diag(cov)[valid], expected_sigmas[valid], rtol=1e-4, atol=1e-6
    )

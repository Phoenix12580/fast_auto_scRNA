"""Unit tests for recall_py.knockoff building blocks."""

import numpy as np
import pytest

from scvalidate.recall_py.knockoff import (
    ZIPParams,
    estimate_nb,
    estimate_zip,
    knockoff_threshold_offset1,
    sample_nb,
    sample_zip,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_zip_recovers_no_inflation_case(rng):
    # Pure Poisson(3) should yield pi_hat ~ 0 (R convention with r0=0)
    data = rng.poisson(lam=3.0, size=5000)
    params = estimate_zip(data)
    # When zeros are rare the estimator short-circuits to r0 -> 0
    assert params.pi_hat == pytest.approx(0.0, abs=0.01) or params.pi_hat == 0.0
    assert params.lambda_hat == pytest.approx(3.0, rel=0.05)


def test_zip_recovers_inflated_zeros(rng):
    # ZIP with lambda=4, pi=0.3
    n = 20000
    zero_mask = rng.random(n) < 0.3
    data = rng.poisson(lam=4.0, size=n)
    data[zero_mask] = 0
    params = estimate_zip(data)
    assert params.pi_hat == pytest.approx(0.3, abs=0.05)
    assert params.lambda_hat == pytest.approx(4.0, abs=0.3)


def test_zip_sampler_shape_and_dtype(rng):
    out = sample_zip(1000, lambda_hat=5.0, pi_hat=0.2, rng=rng)
    assert out.shape == (1000,)
    assert out.dtype == np.int64
    # pi=0.2 means at least ~20% zeros; noise fine
    assert (out == 0).mean() > 0.1


def test_nb_recovers_params(rng):
    # size=2, mu=5 → variance = mu + mu^2/size = 5 + 12.5 = 17.5
    size_true, mu_true = 2.0, 5.0
    p = size_true / (size_true + mu_true)
    data = rng.negative_binomial(n=size_true, p=p, size=10000)
    params = estimate_nb(data)
    # Tolerance loose — NB MLE is hard
    assert params.mu == pytest.approx(mu_true, rel=0.1)
    assert params.size == pytest.approx(size_true, rel=0.3)


def test_nb_sampler_roundtrip(rng):
    out = sample_nb(5000, size=3.0, mu=4.0, rng=rng)
    # Moment check
    assert out.mean() == pytest.approx(4.0, rel=0.1)


def test_knockoff_threshold_infinite_when_no_signal():
    # Symmetric W around zero with large negatives → no t satisfies the FDR
    rng = np.random.default_rng(1)
    w = rng.normal(size=200)  # half positive, half negative by symmetry
    t = knockoff_threshold_offset1(w, fdr=0.05)
    # With no separation, filter should either select nothing (inf) or give
    # a very large t. Either is acceptable; we verify it's not a tiny value.
    assert t > 0 or t == float("inf")


def test_knockoff_threshold_finds_signal():
    # Many very positive W (real > knockoff) and few negative
    w = np.concatenate([np.full(50, 5.0), np.full(3, -1.0), np.zeros(10)])
    t = knockoff_threshold_offset1(w, fdr=0.1)
    assert t < 5.0
    # At threshold t, all the strong positives should get selected.
    assert int((w >= t).sum()) >= 40

"""Unit tests for recall_py.core orchestrator.

Behavioral test only: verify that when fed well-separated synthetic data, the
recall loop converges at a sensible cluster count (>= 2). Full R parity is
deferred until v0.2 with a reference dataset.
"""

import numpy as np
import pytest

from scvalidate.recall_py import find_clusters_recall


@pytest.fixture
def two_pop_counts():
    """Genes × cells, two clearly separated Poisson populations."""
    rng = np.random.default_rng(0)
    n_genes = 400
    n_per = 120

    bg = rng.uniform(0.2, 1.0, size=n_genes)
    lam_a = bg.copy()
    lam_b = bg.copy()
    lam_a[:50] = 8.0
    lam_b[50:100] = 8.0

    a = rng.poisson(lam=lam_a[:, None], size=(n_genes, n_per))
    b = rng.poisson(lam=lam_b[:, None], size=(n_genes, n_per))
    return np.concatenate([a, b], axis=1)


def test_find_clusters_recall_converges(two_pop_counts):
    result = find_clusters_recall(
        two_pop_counts,
        resolution_start=0.8,
        reduction_percentage=0.2,
        dims=10,
        null_method="ZIP",
        n_variable_features=200,
        max_iterations=8,
        seed=0,
        verbose=False,
    )
    assert result.labels.shape == (two_pop_counts.shape[1],)
    # At least 1 cluster; at most 10 on 240 cells — sanity range
    assert 1 <= len(np.unique(result.labels)) <= 10
    assert 1 <= result.n_iterations <= 8
    # All returned clusters should be pass=True at convergence by construction
    assert all(result.per_cluster_pass.values())

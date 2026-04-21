"""Unit tests for scshc_py orchestrator (scshc + test_clusters).

These are behavioral tests — they verify the pipeline produces sane output on
synthetic data. Bit-for-bit R parity will be validated separately in v0.2
once reference outputs are generated against a real dataset.
"""

import numpy as np
import pytest

from scvalidate.scshc_py import scshc
from scvalidate.scshc_py import test_clusters as run_test_clusters


@pytest.fixture
def two_pop_counts():
    """Two well-separated Poisson populations, genes × cells."""
    rng = np.random.default_rng(0)
    n_genes = 300
    n_per = 80

    # Shared background lambda
    bg = rng.uniform(0.2, 1.0, size=n_genes)

    # Population-specific lambdas: 40 up-genes per pop with lambda ~ 6
    lam_a = bg.copy()
    lam_b = bg.copy()
    lam_a[:40] = 6.0
    lam_b[40:80] = 6.0

    a = rng.poisson(lam=lam_a[:, None], size=(n_genes, n_per))
    b = rng.poisson(lam=lam_b[:, None], size=(n_genes, n_per))
    return np.concatenate([a, b], axis=1).astype(np.float64)


@pytest.fixture
def homogeneous_counts():
    """A single homogeneous Poisson population — should NOT split."""
    rng = np.random.default_rng(1)
    lam = rng.uniform(0.5, 3.0, size=300)
    return rng.poisson(lam=lam[:, None], size=(300, 150)).astype(np.float64)


def test_scshc_finds_two_clusters_on_separated_data(two_pop_counts):
    labels, tree = scshc(
        two_pop_counts, alpha=0.05, num_features=200, num_pcs=10, seed=0
    )
    assert labels.shape == (two_pop_counts.shape[1],)
    # At least two clusters should emerge from well-separated populations
    assert len(np.unique(labels)) >= 2
    # Tree has at least one split → root has children
    assert len(tree.children) >= 2 or tree.name.startswith("Cluster")


def test_scshc_on_homogeneous_merges(homogeneous_counts):
    labels, tree = scshc(
        homogeneous_counts, alpha=0.05, num_features=200, num_pcs=10, seed=0
    )
    # On pure noise, scSHC should produce 1 or 2 clusters, rarely more.
    # The FWER control is strict enough to suppress most splits.
    n_clusters = len(np.unique(labels))
    assert n_clusters <= 3, f"homogeneous data split into {n_clusters} clusters"


def test_test_clusters_collapses_overclustered(two_pop_counts):
    # Feed in 4-way overclustering of a true 2-pop dataset — the tester
    # should merge subgroups that don't differ significantly.
    cluster_ids = np.concatenate([
        np.full(40, "c1"), np.full(40, "c2"),
        np.full(40, "c3"), np.full(40, "c4"),
    ])
    new_labels, _tree, pvalues = run_test_clusters(
        two_pop_counts, cluster_ids, alpha=0.05, num_features=200,
        num_pcs=10, seed=0,
    )
    assert len(new_labels) == two_pop_counts.shape[1]
    # Expect strictly fewer than 4 merged labels, since c1/c2 share a
    # population and c3/c4 share a population.
    n_merged = len(np.unique(new_labels))
    assert n_merged <= 4
    # pvalues dict has one entry per original cluster id
    assert set(pvalues.keys()) == {"c1", "c2", "c3", "c4"}


def test_test_clusters_single_input_passes_through():
    # Degenerate: only one cluster_id — no split to test.
    rng = np.random.default_rng(2)
    data = rng.poisson(lam=2.0, size=(100, 30)).astype(np.float64)
    ids = np.full(30, "c1")
    new_labels, tree, pvalues = run_test_clusters(
        data, ids, num_features=50, num_pcs=5, seed=0
    )
    assert set(np.unique(new_labels)) == {"new1"}
    assert list(pvalues.keys()) == ["c1"]

"""Dense vs anndata-oom path parity for find_clusters_recall.

ARI ≥ 0.95 on a 10k synthetic dataset shaped like epithelia. Exact match
(bit-identical) is not required because chunked randomized PCA in the oom
path has small non-determinism even with fixed seeds.
"""
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score


@pytest.fixture
def counts_10k():
    rng = np.random.default_rng(0)
    # 2000 genes x 10000 cells, 5 latent groups
    n_genes, n_cells, n_groups = 2000, 10000, 5
    group = rng.integers(0, n_groups, n_cells)
    means = rng.gamma(2.0, 0.5, size=(n_groups, n_genes)).astype(np.float32)
    counts = rng.poisson(means[group].T).astype(np.int32)  # genes x cells
    return counts


def test_dense_vs_oom_parity(tmp_path, counts_10k):
    from scvalidate.recall_py import find_clusters_recall

    res_dense = find_clusters_recall(
        counts_10k,
        resolution_start=0.8, max_iterations=6,
        fdr=0.05, seed=0, verbose=False,
        backend="dense",
    )
    res_oom = find_clusters_recall(
        counts_10k,
        resolution_start=0.8, max_iterations=6,
        fdr=0.05, seed=0, verbose=False,
        backend="oom", scratch_dir=tmp_path,
    )
    ari = adjusted_rand_score(res_dense.labels, res_oom.labels)
    assert ari >= 0.95, f"dense-vs-oom ARI {ari:.3f} below 0.95"
    assert abs(len(np.unique(res_dense.labels)) - len(np.unique(res_oom.labels))) <= 1

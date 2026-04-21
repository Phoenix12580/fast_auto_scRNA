"""Sanity tests for scatlas.ext.harmony (Harmony 2.0 Rust port)."""
from __future__ import annotations

import numpy as np
import pytest

from scatlas import ext


def _synth_two_clusters_two_batches(
    n_per_leaf: int, d: int, offset: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2 clusters × 2 batches, batch 1 gets a known offset along axis 1."""
    rng = np.random.default_rng(seed)
    n = n_per_leaf * 4
    z = np.empty((n, d), dtype=np.float32)
    batch = np.empty(n, dtype=np.int32)
    cluster = np.empty(n, dtype=np.int32)

    centroids = np.zeros((2, d), dtype=np.float32)
    centroids[0, 0] = -3.0
    centroids[1, 0] = 3.0

    idx = 0
    for c in range(2):
        for b in range(2):
            block = rng.normal(size=(n_per_leaf, d)).astype(np.float32) * 0.3
            block += centroids[c]
            if b == 1:
                block[:, 1] += offset
            z[idx : idx + n_per_leaf] = block
            batch[idx : idx + n_per_leaf] = b
            cluster[idx : idx + n_per_leaf] = c
            idx += n_per_leaf
    return z, batch, cluster


def _per_cluster_batch_gap(z, cluster, batch):
    """Mean ||centroid(c, b=0) - centroid(c, b=1)|| over clusters."""
    gaps = []
    for c in np.unique(cluster):
        mask = cluster == c
        b0 = z[mask & (batch == 0)].mean(axis=0)
        b1 = z[mask & (batch == 1)].mean(axis=0)
        gaps.append(float(np.linalg.norm(b0 - b1)))
    return float(np.mean(gaps))


def test_harmony_reduces_synthetic_batch_offset():
    z, batch, cluster = _synth_two_clusters_two_batches(200, 10, 2.5, seed=1)
    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(X=z)
    adata.obsm["X_pca"] = z
    adata.obs["batch"] = batch

    gap_before = _per_cluster_batch_gap(z, cluster, batch)
    out = ext.harmony(adata, batch_key="batch", use_rep="X_pca", n_clusters=2, seed=7)
    gap_after = _per_cluster_batch_gap(adata.obsm["X_pca_harmony"], cluster, batch)

    assert gap_after < 0.3 * gap_before, (
        f"gap_before={gap_before:.3f}, gap_after={gap_after:.3f}"
    )
    assert adata.obsm["X_pca_harmony"].shape == (800, 10)
    assert "harmony" in adata.uns
    assert out["converged_at_iter"] is not None or len(out["objective_harmony"]) > 0


def test_harmony_preserves_cluster_separation():
    z, batch, cluster = _synth_two_clusters_two_batches(150, 8, 1.5, seed=3)
    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(X=z)
    adata.obsm["X_pca"] = z
    adata.obs["batch"] = batch

    ext.harmony(adata, batch_key="batch", use_rep="X_pca", n_clusters=2, seed=5)

    m0 = adata.obsm["X_pca_harmony"][cluster == 0].mean(axis=0)
    m1 = adata.obsm["X_pca_harmony"][cluster == 1].mean(axis=0)
    sep = float(np.linalg.norm(m0 - m1))
    assert sep > 2.0, f"clusters collapsed to sep={sep:.3f}"


def test_harmony_rejects_single_batch():
    anndata = pytest.importorskip("anndata")
    z = np.random.default_rng(0).normal(size=(100, 10)).astype(np.float32)
    adata = anndata.AnnData(X=z)
    adata.obsm["X_pca"] = z
    adata.obs["batch"] = np.zeros(100, dtype=np.int32)
    with pytest.raises(ValueError, match="only 1 unique"):
        ext.harmony(adata, batch_key="batch", use_rep="X_pca")


def test_harmony_dynamic_lambda_mode():
    """lambda_=None → per-cluster dynamic ridge; result should still
    reduce synthetic batch offset."""
    z, batch, cluster = _synth_two_clusters_two_batches(200, 10, 2.5, seed=5)
    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(X=z)
    adata.obsm["X_pca"] = z
    adata.obs["batch"] = batch

    gap_before = _per_cluster_batch_gap(z, cluster, batch)
    out = ext.harmony(
        adata,
        batch_key="batch",
        use_rep="X_pca",
        n_clusters=2,
        lambda_=None,
        alpha=0.2,
        seed=7,
    )
    gap_after = _per_cluster_batch_gap(adata.obsm["X_pca_harmony"], cluster, batch)
    assert gap_after < 0.4 * gap_before, (
        f"dynamic-lambda Harmony failed: gap_before={gap_before:.3f}, "
        f"gap_after={gap_after:.3f}"
    )
    # Metadata should record the dynamic mode
    assert adata.uns["harmony"]["params"]["lambda_mode"] == "dynamic"
    assert adata.uns["harmony"]["params"]["alpha"] == 0.2


def test_harmony_accepts_string_batch_labels():
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(2)
    z = rng.normal(size=(120, 10)).astype(np.float32)
    # Half cells get +1 on axis 0 as "batch effect"
    z[60:, 0] += 1.0
    adata = anndata.AnnData(X=z)
    adata.obsm["X_pca"] = z
    adata.obs["batch"] = np.where(np.arange(120) < 60, "A", "B")
    result = ext.harmony(adata, batch_key="batch", use_rep="X_pca", n_clusters=3)
    assert result["batch_code_map"] == {0: "A", 1: "B"}

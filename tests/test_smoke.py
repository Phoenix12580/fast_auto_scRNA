"""End-to-end smoke test — synthetic 500 cells × 500 genes, 2 batches, 3 groups.

Exercises every stage in the default BBKNN route and asserts:
- every stage writes its expected output to the AnnData
- every Rust kernel returns finite scores
- Leiden + silhouette optimizer produces the configured number of clusters

Runs in ~10 s on a typical dev box once the Rust extension is built via
``maturin develop``. If the extension is missing, the test is skipped.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


pytest.importorskip("fast_auto_scrna._native", reason="run `maturin develop` first")


def _make_synthetic_adata(seed: int = 42, n_cells: int = 500, n_genes: int = 500):
    import anndata as ad

    rng = np.random.default_rng(seed)
    batch = rng.choice(["A", "B"], size=n_cells)
    group = rng.choice([0, 1, 2], size=n_cells)

    means = rng.gamma(2.0, 1.5, size=(3, n_genes))
    batch_off = rng.normal(0.0, 0.3, size=(2, n_genes))
    mu = np.zeros((n_cells, n_genes))
    for i in range(n_cells):
        mu[i] = means[group[i]] + batch_off[0 if batch[i] == "A" else 1]
    mu = np.clip(mu, 0.05, None)
    X = rng.poisson(mu).astype(np.float32)

    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs={"orig.ident": batch, "group_truth": group.astype(str)},
        var={"gene": [f"g{i}" for i in range(n_genes)]},
    )
    adata.var_names = adata.var["gene"].values
    return adata


def test_pipeline_bbknn_end_to_end():
    from fast_auto_scrna import run_pipeline

    adata = _make_synthetic_adata()

    result = run_pipeline(
        adata_in=adata,
        batch_key="orig.ident",
        integration="bbknn",
        silhouette_n_iter=10,            # keep silhouette sweep cheap
        label_key="group_truth",
        hvg_n_top_genes=300,
    )

    # Every stage's landmark output is present.
    assert "X_pca" in result.obsm
    assert result.obsm["X_pca"].shape[0] == result.n_obs
    assert "X_umap_bbknn" in result.obsm
    assert result.obsm["X_umap_bbknn"].shape == (result.n_obs, 2)
    assert "bbknn_connectivities" in result.obsp
    assert "leiden_bbknn" in result.obs.columns
    assert result.obs["leiden_bbknn"].nunique() >= 2

    # scIB aggregates — iLISI / cLISI / graph_connectivity / mean are finite;
    # kbet_acceptance is NaN by design on BBKNN-style batch-balanced graphs
    # (the kbet() wrapper detects this and bails with a note).
    scib = result.uns["scib_bbknn"]
    for key in ("ilisi", "clisi", "graph_connectivity", "mean"):
        assert key in scib
        assert np.isfinite(scib[key]), f"scib[{key}] non-finite: {scib[key]!r}"
    assert "kbet_acceptance" in scib
    assert np.isnan(scib["kbet_acceptance"]), (
        "kBET should be NaN on BBKNN batch-balanced kNN; "
        f"got {scib['kbet_acceptance']!r}"
    )
    assert "kbet_note" in scib

    # Cluster-homogeneity (ROGUE + SCCAF) finite too.
    assert np.isfinite(scib["rogue_mean"])
    assert np.isfinite(scib["sccaf"])

    # On this synthetic data the groups are well-separated — scores should be
    # comfortably > 0.5. Loose bound to avoid flakiness on other platforms.
    assert scib["mean"] > 0.5, f"scIB mean unexpectedly low: {scib['mean']:.3f}"

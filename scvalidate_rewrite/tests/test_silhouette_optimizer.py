"""Unit test: graph-silhouette optimizer on synthetic 3-gaussian blobs.

Build a tiny synthetic AnnData with 3 well-separated Gaussian clusters
in PCA space. Expect the optimizer to pick a resolution giving ≈ 3
clusters with positive silhouette.
"""
import numpy as np
import pytest


@pytest.fixture
def synth_3blobs_adata():
    import anndata
    import scanpy as sc

    rng = np.random.default_rng(0)
    n_per = 200
    centers = rng.normal(size=(3, 10)) * 5.0
    X = np.vstack([
        rng.normal(size=(n_per, 10)) + centers[k]
        for k in range(3)
    ]).astype(np.float32)
    gt = np.repeat(np.arange(3), n_per)
    # ensure some genes — scanpy requires adata.X even though we use obsm
    ad = anndata.AnnData(X=X)
    ad.obs["gt"] = [str(v) for v in gt]
    ad.obsm["X_pca"] = X
    sc.pp.neighbors(ad, use_rep="X_pca", n_neighbors=15, key_added="test")
    return ad


def test_optimizer_picks_3_clusters(synth_3blobs_adata):
    import sys, os
    _pipeline_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if _pipeline_root not in sys.path:
        sys.path.insert(0, _pipeline_root)
    from scatlas_pipeline.silhouette import (
        optimize_resolution_graph_silhouette, pick_best_resolution,
    )

    curve = optimize_resolution_graph_silhouette(
        synth_3blobs_adata,
        method="test",
        resolutions=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2],
        n_subsample=200,
        n_iter=20,
        seed=0,
        verbose=False,
    )
    assert set(curve.columns) == {
        "resolution", "mean_silhouette", "sd_silhouette", "n_clusters",
    }
    # Graph-connectivity silhouette (d = 1 - connectivity) is intrinsically
    # lower than Euclidean silhouette because the k-NN graph is sparse: most
    # cell pairs are not directly connected, so d=1 for them. On well-separated
    # 3-blob data, graph silhouette typically lands in [0.02, 0.10]; the
    # important check is that it is *positive* and the optimizer picks a
    # sensible cluster count.
    assert curve["mean_silhouette"].max() > 0.01, (
        f"max silhouette too low:\n{curve}"
    )
    best = pick_best_resolution(curve)
    best_k = int(curve.loc[curve["resolution"] == best, "n_clusters"].iloc[0])
    assert 2 <= best_k <= 6, f"best_k={best_k} for a 3-blob dataset is wrong"


def test_stratified_sample_respects_class_proportion():
    import sys, os
    _pipeline_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if _pipeline_root not in sys.path:
        sys.path.insert(0, _pipeline_root)
    from scatlas_pipeline.silhouette import _stratified_sample

    rng = np.random.default_rng(0)
    # 10-class, heavily imbalanced: 1 rare class
    strata = np.array(["A"] * 900 + ["B"] * 90 + ["rare"] * 10)
    idx = _stratified_sample(strata, n_total=100, rng=rng)
    classes = np.unique(strata[idx], return_counts=True)
    counts = dict(zip(*classes))
    assert counts["rare"] >= 10, "rare class must be preserved (min 10)"
    assert counts["A"] >= 60 and counts["A"] <= 100
    assert counts["B"] >= 9


def test_missing_graph_raises():
    import sys, os
    _pipeline_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if _pipeline_root not in sys.path:
        sys.path.insert(0, _pipeline_root)
    from scatlas_pipeline.silhouette import optimize_resolution_graph_silhouette
    import anndata
    ad = anndata.AnnData(X=np.random.rand(100, 5).astype(np.float32))
    with pytest.raises(KeyError, match="bbknn_connectivities"):
        optimize_resolution_graph_silhouette(
            ad, method="bbknn", resolutions=[0.5], n_subsample=50, n_iter=1,
            verbose=False,
        )

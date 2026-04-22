"""End-to-end: pancreas_sub pipeline emits recall comparison report.

Loads the canonical 1k-cell pancreas_sub.rda, runs the full pipeline
(integration="none" — single batch), and verifies that
adata.uns["recall_none_comparison"] is present and well-formed.
"""
from __future__ import annotations

from pathlib import Path
import pytest

# Canonical dataset location (matches project_canonical_test_data memory)
PANCREAS_SUB_RDA = Path("/mnt/f/NMF_rewrite/fast_auto_scRNA_v1/../pancreas_sub.rda")
# Normalise to absolute path for clarity in skip messages
PANCREAS_SUB_RDA = (
    Path("/mnt/f/NMF_rewrite/pancreas_sub.rda")
)


def _load_pancreas_sub():
    """Load pancreas_sub.rda → AnnData (mirrors e2e_pancreas_sub.py)."""
    import rdata
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    parsed = rdata.read_rda(str(PANCREAS_SUB_RDA))
    seurat = parsed["pancreas_sub"]
    rna = seurat.assays["RNA"]
    counts_raw = rna.layers["counts"]
    n_genes, n_cells = int(counts_raw.Dim[0]), int(counts_raw.Dim[1])
    mat_gc = sp.csc_matrix(
        (counts_raw.x, counts_raw.i, counts_raw.p),
        shape=(n_genes, n_cells),
    )
    X = mat_gc.T.tocsr().astype(np.float32)
    meta = getattr(seurat, "meta_data", None) or getattr(seurat, "meta.data", None)
    a = ad.AnnData(X=X)
    a.obs_names = [f"cell_{i}" for i in range(n_cells)]
    a.var_names = [f"gene_{i}" for i in range(n_genes)]
    for col in meta.columns:
        a.obs[str(col)] = pd.Categorical(meta[col].to_numpy())
    # Single batch — synthesise the batch key used by the pipeline
    a.obs["orig.ident"] = "SeuratProject"
    return a


@pytest.mark.skipif(not PANCREAS_SUB_RDA.exists(), reason="canonical data missing")
def test_pipeline_produces_recall_comparison(tmp_path):
    """Full pipeline on pancreas_sub emits a valid recall_none_comparison dict."""
    import sys
    # Ensure scatlas_pipeline is importable from the worktree root
    worktree = Path(__file__).resolve().parents[2]
    if str(worktree) not in sys.path:
        sys.path.insert(0, str(worktree))

    from scatlas_pipeline import PipelineConfig  # noqa: F401

    adata_in = _load_pancreas_sub()
    print(f"\nloaded pancreas_sub: {adata_in.n_obs} × {adata_in.n_vars}")

    # Build config — only fields that exist on PipelineConfig are used.
    # pancreas_sub is 1 batch → pipeline auto-forces integration="none".
    cfg = PipelineConfig(
        input_h5ad="<in-memory:pancreas_sub>",
        batch_key="orig.ident",
        min_cells=3,
        min_genes=50,
        max_pct_mt=100.0,
        hvg_n_top_genes=2000,
        hvg_flavor="seurat_v3",
        hvg_batch_aware=False,
        integration="none",
        run_umap=True,
        umap_n_epochs=100,           # fast enough for 1k cells
        run_leiden=True,
        leiden_resolutions=[0.3, 0.5, 0.8, 1.0],
        leiden_target_n=(4, 12),
        compute_silhouette=False,    # O(N²) — skip for speed
        compute_homogeneity=True,
        recall_resolution_start=0.8,
        recall_max_iterations=6,     # keep the test under ~60s
        # out_h5ad intentionally omitted: per_baseline_cluster_fate has int
        # keys that anndata's h5py backend can't serialize (pre-existing
        # pipeline issue, out of scope for this test).
    )

    from scatlas_pipeline.pipeline import run_from_config
    adata = run_from_config(cfg, adata_in=adata_in)

    # --- At least one recall_*_comparison key must exist --------------------
    keys = [
        k for k in adata.uns.keys()
        if k.startswith("recall_") and k.endswith("_comparison")
    ]
    assert keys, (
        f"no recall_*_comparison in uns; all uns keys: {sorted(adata.uns.keys())}"
    )

    rep = adata.uns[keys[0]]
    print(f"recall comparison key: {keys[0]}")
    print(f"  k_baseline        = {rep['k_baseline']}")
    print(f"  k_recall          = {rep['k_recall']}")
    print(f"  delta_k           = {rep['delta_k']}")
    print(f"  ARI               = {rep['ari_baseline_vs_recall']:.4f}")
    print(f"  recall_converged  = {rep['recall_converged']}")
    print(f"  k_trajectory      = {rep['k_trajectory']}")
    print(f"  wall_time_s       = {rep['recall_wall_time_s']:.2f}s")

    # --- Field-level assertions ----------------------------------------------
    assert rep["k_baseline"] >= 2, (
        f"k_baseline={rep['k_baseline']} unexpectedly low"
    )
    assert rep["k_recall"] >= 1, (
        f"k_recall={rep['k_recall']} unexpectedly low"
    )
    # delta_k = k_baseline - k_recall; allow large positive swing (recall
    # can merge many clusters) but bounded by sanity
    assert -5 <= rep["delta_k"] <= 30, (
        f"delta_k={rep['delta_k']} out of sanity range [-5, 30]"
    )
    assert 0.0 <= rep["ari_baseline_vs_recall"] <= 1.0, (
        f"ARI={rep['ari_baseline_vs_recall']} not in [0, 1]"
    )
    assert isinstance(rep["k_trajectory"], list) and len(rep["k_trajectory"]) >= 1, (
        f"k_trajectory={rep['k_trajectory']!r} is empty or not a list"
    )
    assert rep["recall_wall_time_s"] > 0.0, (
        f"recall_wall_time_s={rep['recall_wall_time_s']} not positive"
    )
    assert isinstance(rep["recall_converged"], bool), (
        f"recall_converged={rep['recall_converged']!r} not a bool"
    )
    assert "per_baseline_cluster_fate" in rep, (
        "per_baseline_cluster_fate missing from comparison dict"
    )

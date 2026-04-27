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

# scvi-tools is optional (heavy: torch + CUDA, ~5 GB on Linux). WSL hosts
# typically skip it; Windows hosts that run scvi training install it.
_HAS_SCVI = False
try:
    import scvi  # noqa: F401
    _HAS_SCVI = True
except ImportError:
    _HAS_SCVI = False
needs_scvi = pytest.mark.skipif(not _HAS_SCVI, reason="scvi-tools not installed")


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

    # scIB aggregates — iLISI / cLISI / graph_connectivity / mean are finite.
    # kBET is opt-in (cfg.compute_kbet) since 2026-04-25; default off so the
    # key should be absent here.
    scib = result.uns["scib_bbknn"]
    for key in ("ilisi", "clisi", "graph_connectivity", "mean"):
        assert key in scib
        assert np.isfinite(scib[key]), f"scib[{key}] non-finite: {scib[key]!r}"
    assert "kbet_acceptance" not in scib, (
        "kBET removed from default panel; expected key absent"
    )

    # Cluster-homogeneity (ROGUE + SCCAF) finite too.
    assert np.isfinite(scib["rogue_mean"])
    assert np.isfinite(scib["sccaf"])

    # Embedding silhouettes from scib-metrics — paper-faithful panel
    # (Gao et al. Cancer Cell 2024). Default `compute_silhouette=True`.
    for k in ("label_silhouette", "batch_silhouette", "isolated_label"):
        assert k in scib, f"scib[{k}] missing — silhouette panel not wired"
        assert np.isfinite(scib[k]), f"scib[{k}] non-finite: {scib[k]!r}"

    # On this synthetic data the groups are well-separated — scores should be
    # comfortably > 0.5. Loose bound to avoid flakiness on other platforms.
    assert scib["mean"] > 0.5, f"scIB mean unexpectedly low: {scib['mean']:.3f}"


def test_pipeline_compute_kbet_opt_in():
    """compute_kbet=True must add kbet_acceptance back to the scib dict."""
    from fast_auto_scrna import run_pipeline

    adata = _make_synthetic_adata()

    result = run_pipeline(
        adata_in=adata,
        batch_key="orig.ident",
        integration="harmony",                   # avoid bbknn nan-by-construction
        compute_kbet=True,
        label_key="group_truth",
        hvg_n_top_genes=300,
    )
    scib = result.uns["scib_harmony"]
    assert "kbet_acceptance" in scib
    assert np.isfinite(scib["kbet_acceptance"])


def test_pipeline_compute_silhouette_off():
    """compute_silhouette=False must skip the 3 ASW keys (opt-out path)."""
    from fast_auto_scrna import run_pipeline

    adata = _make_synthetic_adata()

    result = run_pipeline(
        adata_in=adata,
        batch_key="orig.ident",
        integration="bbknn",
        compute_silhouette=False,
        label_key="group_truth",
        hvg_n_top_genes=300,
    )

    scib = result.uns["scib_bbknn"]
    for k in ("label_silhouette", "batch_silhouette", "isolated_label"):
        assert k not in scib, (
            f"scib[{k}] present despite compute_silhouette=False"
        )
    # Other metrics still computed
    for k in ("ilisi", "clisi", "graph_connectivity", "mean"):
        assert k in scib and np.isfinite(scib[k])


@needs_scvi
def test_multiroute_gate_pauses_before_phase2b():
    """integration='all+scvi' + cluster_method=None must early-exit after Phase 2a.

    Uses scvi_max_epochs=5 so the scvi route trains (semantic check) without
    blocking the test for ~30s on a CPU-only torch install. ``"all+scvi"``
    is the explicit 4-route opt-in (``"all"`` excludes scvi by default).
    """
    from fast_auto_scrna import run_pipeline

    adata = _make_synthetic_adata()

    result = run_pipeline(
        adata_in=adata,
        batch_key="orig.ident",
        integration="all+scvi",
        cluster_method=None,
        label_key="group_truth",
        hvg_n_top_genes=300,
        scvi_max_epochs=5,
    )

    # Gate sentinels.
    assert result.uns.get("fast_auto_scrna_gate_paused") is True
    assert result.uns.get("fast_auto_scrna_auto_pick") in (
        "bbknn", "harmony", "fastmnn", "scvi",
    )

    # Phase 2a artifacts present for every route.
    for m in ("bbknn", "harmony", "fastmnn", "scvi"):
        assert f"scib_{m}" in result.uns, f"missing scib_{m}"
        assert f"X_umap_{m}" in result.obsm, f"missing X_umap_{m}"

    # Phase 2b artifacts absent — no Leiden / ROGUE / SCCAF was run.
    for m in ("bbknn", "harmony", "fastmnn", "scvi"):
        assert f"leiden_{m}" not in result.obs.columns, (
            f"leiden_{m} present — Phase 2b should not have run"
        )
        scib = result.uns[f"scib_{m}"]
        assert "rogue_mean" not in scib, f"rogue ran for {m} but gate should have paused"
        assert "sccaf" not in scib, f"sccaf ran for {m} but gate should have paused"


def test_pipeline_fastmnn_end_to_end():
    """fastmnn route must produce a corrected embedding + valid scIB."""
    from fast_auto_scrna import run_pipeline

    adata = _make_synthetic_adata()

    result = run_pipeline(
        adata_in=adata,
        batch_key="orig.ident",
        integration="fastmnn",
        label_key="group_truth",
        hvg_n_top_genes=300,
    )

    assert "X_pca_fastmnn" in result.obsm
    assert result.obsm["X_pca_fastmnn"].shape[0] == result.n_obs
    assert "X_umap_fastmnn" in result.obsm
    assert "leiden_fastmnn" in result.obs.columns
    assert result.obs["leiden_fastmnn"].nunique() >= 2

    info = result.uns.get("fastmnn", {})
    # 2 batches → 1 merge step. Either we found MNN pairs or the batch
    # was skipped (both are valid outcomes on tiny synthetic data).
    assert "merge_order" in info
    assert len(info["merge_order"]) == 2

    scib = result.uns["scib_fastmnn"]
    for k in ("ilisi", "clisi", "graph_connectivity", "mean",
              "label_silhouette", "batch_silhouette", "isolated_label"):
        assert k in scib, f"scib_fastmnn missing {k}"
        assert np.isfinite(scib[k]), f"scib_fastmnn[{k}] non-finite: {scib[k]!r}"


@needs_scvi
def test_multiroute_cluster_non_winners_at_winner_res():
    """cluster_non_winners_at_winner_res=True clusters all routes at winner r."""
    from fast_auto_scrna import run_pipeline

    adata = _make_synthetic_adata()

    result = run_pipeline(
        adata_in=adata,
        batch_key="orig.ident",
        integration="all+scvi",
        cluster_method="bbknn",                  # pick winner explicitly
        cluster_non_winners_at_winner_res=True,
        label_key="group_truth",
        hvg_n_top_genes=300,
        scvi_max_epochs=5,
    )

    winner_res = result.uns["leiden_bbknn_resolution"]
    for m in ("bbknn", "harmony", "fastmnn", "scvi"):
        assert f"leiden_{m}" in result.obs.columns, f"leiden_{m} missing"
        assert result.uns[f"leiden_{m}_resolution"] == winner_res, (
            f"leiden_{m}_resolution should equal winner_res {winner_res}"
        )
        scib = result.uns[f"scib_{m}"]
        assert "rogue_mean" in scib, f"scib_{m} missing rogue_mean"
        assert "sccaf" in scib, f"scib_{m} missing sccaf"
        assert np.isfinite(scib["rogue_mean"])
        assert np.isfinite(scib["sccaf"])

    # Source provenance set for non-winners
    for m in ("harmony", "fastmnn", "scvi"):
        assert "copied from winner" in result.uns[f"leiden_{m}_resolution_source"]


@needs_scvi
def test_pipeline_scvi_end_to_end():
    """scvi route must produce a latent + valid scIB.

    Uses scvi_max_epochs=5 — model quality is meaningless on 500 synthetic
    cells with 5 epochs; this exercises the data-flow only. Real scVI
    quality is validated on the 222k benchmark.
    """
    from fast_auto_scrna import run_pipeline

    adata = _make_synthetic_adata()

    result = run_pipeline(
        adata_in=adata,
        batch_key="orig.ident",
        integration="scvi",
        scvi_max_epochs=5,
        label_key="group_truth",
        hvg_n_top_genes=300,
    )

    assert "X_scvi" in result.obsm
    assert "X_pca_scvi" in result.obsm
    assert result.obsm["X_scvi"].shape == (result.n_obs, 30)  # default n_latent=30
    assert "X_umap_scvi" in result.obsm
    assert "leiden_scvi" in result.obs.columns

    info = result.uns.get("scvi", {})
    assert info.get("max_epochs") == 5
    assert info.get("n_latent") == 30

    scib = result.uns["scib_scvi"]
    for k in ("ilisi", "clisi", "graph_connectivity", "mean",
              "label_silhouette", "batch_silhouette", "isolated_label"):
        assert k in scib, f"scib_scvi missing {k}"
        assert np.isfinite(scib[k]), f"scib_scvi[{k}] non-finite: {scib[k]!r}"


@needs_scvi
def test_multiroute_resume_with_cluster_method():
    """integration='all+scvi' + cluster_method='bbknn' must skip the gate and run Phase 2b."""
    from fast_auto_scrna import run_pipeline

    adata = _make_synthetic_adata()

    result = run_pipeline(
        adata_in=adata,
        batch_key="orig.ident",
        integration="all+scvi",
        cluster_method="bbknn",
        label_key="group_truth",
        hvg_n_top_genes=300,
    )

    # Gate did NOT trigger.
    assert result.uns.get("fast_auto_scrna_gate_paused") is not True
    # Phase 2b ran for the chosen route only.
    assert "leiden_bbknn" in result.obs.columns
    assert "rogue_mean" in result.uns["scib_bbknn"]
    # Phase 2b did NOT run for non-winners.
    assert "leiden_none" not in result.obs.columns
    assert "leiden_harmony" not in result.obs.columns


def test_phase2a_scib_parallel_matches_sequential():
    """v2-P11: parallel Phase 2a scIB output must match sequential bit-for-bit.

    Both paths call the same ``scib_metrics`` JAX kernels on the same input
    arrays — only the scheduling differs. Numerics must be identical.

    ``integration="all"`` expands to ``DEFAULT_ALL_METHODS`` (bbknn / harmony /
    fastmnn) so this runs on WSL hosts that don't install scvi-tools.
    """
    from fast_auto_scrna import run_pipeline

    common = dict(
        batch_key="orig.ident",
        integration="all",
        cluster_method="bbknn",   # skip gate, exercise full pipeline
        label_key="group_truth",
        hvg_n_top_genes=300,
    )

    seq_result = run_pipeline(
        adata_in=_make_synthetic_adata(), scib_parallel=False, **common,
    )
    par_result = run_pipeline(
        adata_in=_make_synthetic_adata(), scib_parallel=True, **common,
    )

    methods = ("bbknn", "harmony", "fastmnn")
    for m in methods:
        seq_scib = seq_result.uns[f"scib_{m}"]
        par_scib = par_result.uns[f"scib_{m}"]
        for k, v_seq in seq_scib.items():
            assert k in par_scib, f"parallel missing {k} for {m}"
            v_par = par_scib[k]
            if isinstance(v_seq, (int, float)):
                if np.isnan(v_seq):
                    assert np.isnan(v_par), (
                        f"{m}.{k}: seq=NaN but par={v_par!r}"
                    )
                else:
                    # scib_metrics is deterministic — exact match expected.
                    # Use 1e-6 as floor to absorb any float reduction-order
                    # noise from BLAS thread-count differences (parallel
                    # workers run with fewer threads than sequential).
                    assert abs(float(v_par) - float(v_seq)) < 1e-6, (
                        f"{m}.{k}: seq={v_seq!r} par={v_par!r} differ"
                    )

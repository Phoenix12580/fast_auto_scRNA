"""scatlas_pipeline — one-call end-to-end scRNA-seq atlas pipeline.

Load → QC → lognorm → PCA (GD auto) → BBKNN → [Harmony] → UMAP →
Leiden (auto resolution) → [recall-validated clustering] → scib metrics.

Every step that has a Rust-backed kernel in `scatlas` or `scvalidate_rust`
is wired to use it. Python/scipy is used only for IO and arithmetic that's
already fast enough (normalization, CSR construction).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration. Fields map 1:1 to pipeline steps."""

    # --- Input
    input_h5ad: str
    batch_key: str = "orig.ident"
    out_h5ad: str | None = None

    # --- QC (01)
    min_cells: int = 3                    # genes in < min_cells cells → drop
    min_genes: int = 200                  # cells with < min_genes genes → drop
    max_pct_mt: float = 20.0              # cells with MT fraction > pct → drop
    mt_prefix: str = "MT-"

    # --- Normalization (02)
    target_sum: float = 1e4

    # --- PCA (03)
    pca_n_comps: int | str = "auto"       # "auto" → Gavish-Donoho
    pca_n_power_iter: int = 7
    pca_random_state: int = 0

    # --- BBKNN (04) — optional batch-balanced kNN graph integration.
    # When `run_bbknn=False` the pipeline falls back to a standard
    # scanpy kNN on PCA (cosine, k=neighbors_within_batch*2).
    run_bbknn: bool = True
    neighbors_within_batch: int = 3
    bbknn_backend: str = "auto"           # "brute", "hnsw", "auto"

    # --- Harmony (05) — optional embedding-space batch correction
    run_harmony: bool = True
    harmony_n_clusters: int = 100
    harmony_max_iter: int = 10
    harmony_seed: int = 0

    # --- densify OOM guard (optional anndataoom acceleration)
    # When enabled and the `anndataoom` package is installed, densify
    # operations route through it to avoid peak-memory spikes on large
    # sparse → dense conversions.
    use_anndataoom: bool = False

    # --- UMAP (06)
    run_umap: bool = True
    umap_min_dist: float = 0.5
    umap_spread: float = 1.0
    umap_n_epochs: int = 200
    umap_random_state: int = 0

    # --- Leiden (07a)
    run_leiden: bool = True
    leiden_resolutions: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    leiden_target_n: tuple[int, int] = (8, 30)    # pick smallest res in range
    leiden_n_iterations: int = 2

    # --- recall (07b) — optional, scales O(K²) in n_clusters
    run_recall: bool = False
    recall_resolution_start: float = 0.8
    recall_fdr: float = 0.05
    recall_max_iterations: int = 20

    # --- scib metrics (08)
    run_metrics: bool = True
    label_key: str | None = None          # e.g. "subtype"; None → use cluster labels

    # --- ROGUE per-cluster purity (09)
    # Single-cell specific cluster purity metric (Liu 2020, NatComm).
    # Rust kernel via scatlas.stats.calculate_rogue.
    run_rogue: bool = True
    rogue_platform: str = "UMI"           # "UMI" or "full-length"
    rogue_cluster_key: str | None = None  # None → uses 'leiden'


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _banner(s: str) -> None:
    print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")


def _step(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    print(f"  [{label}] {t1 - t0:.1f}s")
    return t1


def run_pipeline(input_h5ad: str | None = None, **kwargs) -> Any:
    """Run the full pipeline. Returns the final AnnData.

    Either pass `input_h5ad` + keyword overrides, or construct a
    `PipelineConfig` and call `run_from_config(cfg)`.
    """
    if input_h5ad is not None:
        cfg = PipelineConfig(input_h5ad=input_h5ad, **kwargs)
    else:
        cfg = PipelineConfig(**kwargs)
    return run_from_config(cfg)


def run_from_config(cfg: PipelineConfig) -> Any:
    import anndata as ad

    _banner(f"scatlas_pipeline — {cfg.input_h5ad}")
    pipeline_t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # --- 00 anndataoom (optional OOM guard) --------------------------------
    _try_enable_anndataoom(cfg)

    # --- 01 load + QC -------------------------------------------------------
    t0 = time.perf_counter()
    adata = ad.read_h5ad(cfg.input_h5ad)
    print(f"[01 load]   raw: {adata.n_obs} × {adata.n_vars}")
    adata = _qc_filter(adata, cfg)
    print(f"         post-QC: {adata.n_obs} × {adata.n_vars}")
    timings["load+qc"] = _step("01 load+qc", t0) - t0

    if adata.obs[cfg.batch_key].dtype == object:
        adata.obs["_batch"] = adata.obs[cfg.batch_key].astype(str)
    else:
        adata.obs["_batch"] = adata.obs[cfg.batch_key].astype(str)
    uniq, cnt = np.unique(adata.obs["_batch"], return_counts=True)
    print(f"         batches: {dict(zip(uniq.tolist(), cnt.tolist()))}")

    # --- 02 lognorm ---------------------------------------------------------
    t0 = time.perf_counter()
    adata.X = _lognorm(adata.X, cfg.target_sum)
    timings["lognorm"] = _step("02 lognorm", t0) - t0

    # --- 03 PCA -------------------------------------------------------------
    t0 = time.perf_counter()
    from scatlas import pp

    pca_out = pp.pca(
        adata,
        n_comps=cfg.pca_n_comps,
        n_power_iter=cfg.pca_n_power_iter,
        random_state=cfg.pca_random_state,
    )
    print(f"         X_pca {adata.obsm['X_pca'].shape}", end="")
    if pca_out.get("auto") is not None:
        info = pca_out["auto"]
        print(f"   GD={info['n_comps_gavish_donoho']} elbow={info['n_comps_elbow']}")
    else:
        print()
    timings["pca"] = _step("03 pca", t0) - t0

    # --- 04 neighbor graph (BBKNN or standard kNN) -------------------------
    t0 = time.perf_counter()
    from scatlas import ext

    if cfg.run_bbknn:
        ext.bbknn(
            adata,
            batch_key="_batch",
            use_rep="X_pca",
            neighbors_within_batch=cfg.neighbors_within_batch,
            with_connectivities=True,
        )
        print(f"         bbknn conn nnz={adata.obsp['bbknn_connectivities'].nnz}")
        timings["bbknn"] = _step("04 bbknn", t0) - t0
    else:
        # Standard scanpy kNN on PCA — no batch-balancing.
        import scanpy as sc
        sc.pp.neighbors(
            adata,
            n_neighbors=cfg.neighbors_within_batch * 2,
            use_rep="X_pca", metric="cosine", random_state=0,
        )
        # Alias scanpy keys to the BBKNN names so downstream (UMAP, leiden)
        # can consume them uniformly.
        adata.obsp["bbknn_connectivities"] = adata.obsp["connectivities"]
        adata.obsp["bbknn_distances"] = adata.obsp["distances"]
        print(f"         standard kNN conn nnz={adata.obsp['bbknn_connectivities'].nnz}")
        timings["neighbors"] = _step("04 neighbors", t0) - t0

    # --- 05 Harmony (optional) ---------------------------------------------
    if cfg.run_harmony:
        t0 = time.perf_counter()
        ext.harmony(
            adata,
            batch_key="_batch",
            use_rep="X_pca",
            n_clusters=cfg.harmony_n_clusters,
            max_iter=cfg.harmony_max_iter,
            seed=cfg.harmony_seed,
        )
        print(f"         X_pca_harmony {adata.obsm['X_pca_harmony'].shape}")
        timings["harmony"] = _step("05 harmony", t0) - t0

    # --- 06 UMAP ------------------------------------------------------------
    if cfg.run_umap:
        t0 = time.perf_counter()
        from scatlas import tl

        tl.umap(
            adata,
            neighbors_key="bbknn",
            min_dist=cfg.umap_min_dist,
            spread=cfg.umap_spread,
            n_epochs=cfg.umap_n_epochs,
            init="pca",
            random_state=cfg.umap_random_state,
        )
        timings["umap"] = _step("06 umap", t0) - t0

    # --- 07a Leiden (auto resolution) --------------------------------------
    if cfg.run_leiden:
        t0 = time.perf_counter()
        labels, chosen_res = _leiden_auto_resolution(adata, cfg)
        adata.obs["leiden"] = labels
        adata.uns["leiden_chosen_resolution"] = chosen_res
        print(f"         picked r={chosen_res} → {len(np.unique(labels))} clusters")
        timings["leiden"] = _step("07a leiden", t0) - t0

    # --- 07b recall-validated clustering (optional) ------------------------
    if cfg.run_recall:
        t0 = time.perf_counter()
        _run_recall_clusters(adata, cfg)
        timings["recall"] = _step("07b recall", t0) - t0

    # --- 08 scib metrics ---------------------------------------------------
    if cfg.run_metrics:
        t0 = time.perf_counter()
        scib = _compute_scib_metrics(adata, cfg)
        adata.uns["scib_score"] = scib
        for k, v in scib.items():
            if isinstance(v, float):
                print(f"             {k:22s} = {v:.3f}")
        timings["metrics"] = _step("08 metrics", t0) - t0

    # --- 09 per-cluster ROGUE ----------------------------------------------
    if cfg.run_rogue and cfg.run_leiden:
        t0 = time.perf_counter()
        _compute_rogue(adata, cfg)
        timings["rogue"] = _step("09 rogue", t0) - t0

    # --- Save ---------------------------------------------------------------
    total = time.perf_counter() - pipeline_t0
    adata.uns["scatlas_pipeline_timings"] = timings
    _banner(f"pipeline done in {total:.1f}s ({total / 60:.1f} min)")
    for k, v in timings.items():
        print(f"  {k:15s} {v:6.1f}s")
    print(f"  {'TOTAL':15s} {total:6.1f}s")

    if cfg.out_h5ad:
        out = Path(cfg.out_h5ad)
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out, compression="lzf")
        print(f"\n  wrote {out}")

    return adata


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def _try_enable_anndataoom(cfg: PipelineConfig) -> bool:
    """Attempt to import and enable the `anndataoom` out-of-memory densifier.

    anndataoom (https://github.com/anndataoom) offers a safer sparse→dense
    path that avoids peak-memory spikes at 1M+ cell scale. When installed
    and enabled via `cfg.use_anndataoom=True`, subsequent AnnData densify
    operations route through it transparently. Returns True if active.
    """
    if not cfg.use_anndataoom:
        return False
    try:
        import anndataoom  # type: ignore  # noqa: F401
        # anndataoom monkey-patches anndata on import
        print("         [anndataoom] active — densify ops routed through OOM guard")
        return True
    except ImportError:
        print("         [anndataoom] requested but not installed — "
              "`pip install anndataoom` to enable")
        return False


def _qc_filter(adata, cfg: PipelineConfig):
    import anndata as ad  # noqa: F401

    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)

    # Genes in < min_cells cells → drop
    gene_cells = (X > 0).sum(axis=0).A1 if sp.issparse(X) else (X > 0).sum(axis=0)
    keep_g = gene_cells >= cfg.min_cells

    # Cells with < min_genes expressed → drop
    cell_genes = (X > 0).sum(axis=1).A1 if sp.issparse(X) else (X > 0).sum(axis=1)
    keep_c = cell_genes >= cfg.min_genes

    # MT fraction
    if cfg.max_pct_mt > 0 and cfg.max_pct_mt < 100:
        mt = np.array(
            [str(v).upper().startswith(cfg.mt_prefix.upper()) for v in adata.var_names]
        )
        if mt.any():
            libs = np.asarray(X.sum(axis=1)).ravel()
            libs_safe = np.where(libs == 0, 1, libs)
            mt_counts = np.asarray(X[:, mt].sum(axis=1)).ravel()
            pct_mt = 100.0 * mt_counts / libs_safe
            keep_c = keep_c & (pct_mt <= cfg.max_pct_mt)

    adata = adata[keep_c, :][:, keep_g].copy()
    return adata


def _lognorm(X, target_sum: float):
    X = X if sp.issparse(X) else sp.csr_matrix(X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((target_sum / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    return Xn.tocsr()


def _leiden_auto_resolution(adata, cfg: PipelineConfig) -> tuple[np.ndarray, float]:
    """Sweep resolutions, return labels at first resolution yielding a
    cluster count inside `cfg.leiden_target_n`. Falls back to middle
    resolution if none hits the target."""
    import scanpy as sc

    # scatlas BBKNN writes bbknn_connectivities — map to scanpy's expected key
    if "connectivities" not in adata.obsp and "bbknn_connectivities" in adata.obsp:
        adata.obsp["connectivities"] = adata.obsp["bbknn_connectivities"]
        adata.obsp["distances"] = adata.obsp["bbknn_distances"]
        adata.uns["neighbors"] = {
            "params": {"method": "umap", "n_neighbors": cfg.neighbors_within_batch * 2},
            "connectivities_key": "connectivities",
            "distances_key": "distances",
        }

    chosen: tuple[float, np.ndarray] | None = None
    for r in cfg.leiden_resolutions:
        sc.tl.leiden(
            adata,
            resolution=r,
            key_added=f"_leiden_r{r}",
            flavor="igraph",
            directed=False,
            n_iterations=cfg.leiden_n_iterations,
            random_state=0,
        )
        n = adata.obs[f"_leiden_r{r}"].nunique()
        in_range = cfg.leiden_target_n[0] <= n <= cfg.leiden_target_n[1]
        print(f"         leiden r={r} → {n} clusters{' ✓' if in_range else ''}")
        if in_range and chosen is None:
            chosen = (r, adata.obs[f"_leiden_r{r}"].to_numpy())
    if chosen is None:
        r = cfg.leiden_resolutions[len(cfg.leiden_resolutions) // 2]
        print(f"         [fallback] no resolution hit target range; r={r}")
        chosen = (r, adata.obs[f"_leiden_r{r}"].to_numpy())
    return chosen[1], chosen[0]


def _run_recall_clusters(adata, cfg: PipelineConfig) -> None:
    """scvalidate's knockoff-validated iterative clustering.
    Only safe at <= ~10k cells — scales O(K² · N).
    """
    try:
        from scvalidate.recall_py import find_clusters_recall
    except ImportError:
        print("         [recall] scvalidate not installed — skipping")
        return
    X = adata.raw.X if adata.raw is not None else adata.X
    # recall expects genes × cells raw counts
    counts_gxc = (X if sp.issparse(X) else sp.csr_matrix(X)).T
    result = find_clusters_recall(
        counts_gxc,
        resolution_start=cfg.recall_resolution_start,
        fdr=cfg.recall_fdr,
        max_iterations=cfg.recall_max_iterations,
        seed=0,
    )
    adata.obs["leiden_recall"] = result.labels.astype(str)
    adata.uns["leiden_recall_resolution"] = result.resolution
    adata.uns["leiden_recall_iterations"] = result.n_iterations
    print(
        f"         recall converged at r={result.resolution:.3f}, "
        f"{result.n_iterations} iters, "
        f"{len(np.unique(result.labels))} clusters"
    )


def _compute_rogue(adata, cfg: PipelineConfig) -> None:
    """Per-cluster ROGUE purity (Liu 2020, NatComm).

    Wraps scvalidate.rogue_py.rogue_per_cluster which internally calls
    the Rust-accelerated entropy_table + calculate_rogue kernels from
    scatlas.stats. High ROGUE (→1) = pure single-state cluster.
    """
    try:
        from scvalidate.rogue_py import rogue_per_cluster
    except ImportError:
        print("         [rogue] scvalidate not installed — skipping")
        return

    cluster_key = cfg.rogue_cluster_key or "leiden"
    if cluster_key not in adata.obs.columns:
        print(f"         [rogue] {cluster_key} missing — skipping")
        return
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    samples = adata.obs["_batch"].astype(str).to_numpy()

    # ROGUE is defined on raw counts. Prefer adata.raw / layers['counts']
    # over the log-normalized adata.X (pipeline overwrote it in-place).
    if adata.raw is not None:
        counts_cxg = adata.raw.X
    elif "counts" in adata.layers:
        counts_cxg = adata.layers["counts"]
    else:
        counts_cxg = adata.X
        print("         [rogue] no raw counts; using current X (approximation)")

    counts_cxg = counts_cxg if sp.issparse(counts_cxg) else sp.csr_matrix(counts_cxg)
    expr_gxc = counts_cxg.T.tocsc()  # rogue_per_cluster expects genes × cells

    try:
        result = rogue_per_cluster(
            expr_gxc, labels=labels, samples=samples,
            platform=cfg.rogue_platform,
        )
    except Exception as e:
        print(f"         [rogue] failed: {type(e).__name__}: {e}")
        return

    mat = result.matrix  # (samples × clusters) DataFrame with NaNs for empty cells
    per_cluster_mean = mat.mean(axis=0, skipna=True).to_dict()
    adata.uns["rogue"] = {
        "cluster_key": cluster_key,
        "platform": cfg.rogue_platform,
        "matrix": mat.to_dict(orient="index"),
        "per_cluster_mean": {str(k): float(v) for k, v in per_cluster_mean.items()
                             if not pd.isna(v)},
    }
    valid = [(str(k), float(v)) for k, v in per_cluster_mean.items() if not pd.isna(v)]
    valid.sort(key=lambda kv: -kv[1])
    if valid:
        overall = float(np.mean([v for _, v in valid]))
        print(f"         [rogue] mean={overall:.3f}, per cluster (top 10):")
        for cid, score in valid[:10]:
            print(f"             cluster {cid:>4s}: ROGUE={score:.3f}")
        if len(valid) > 10:
            print(f"             ... ({len(valid) - 10} more)")
    else:
        print("         [rogue] no clusters had enough cells per sample")


def _compute_scib_metrics(adata, cfg: PipelineConfig):
    from scatlas import ext as sc_ext, metrics as sc_metrics

    raw = sc_ext.bbknn_kneighbors(
        adata.obsm.get("X_pca_harmony", adata.obsm["X_pca"]),
        adata.obs["_batch"].to_numpy(),
        neighbors_within_batch=cfg.neighbors_within_batch,
        backend=cfg.bbknn_backend,
    )
    indices_u32 = raw["indices"]
    sentinel_mask = indices_u32 == np.iinfo(np.uint32).max
    knn_idx = indices_u32.astype(np.int32)
    knn_dist = raw["distances"].astype(np.float32).copy()
    if sentinel_mask.any():
        row_idx = np.broadcast_to(
            np.arange(knn_idx.shape[0], dtype=np.int32)[:, None], knn_idx.shape
        )
        knn_idx = np.where(sentinel_mask, row_idx, knn_idx)
        knn_dist[sentinel_mask] = 0.0

    label_key = cfg.label_key or ("leiden" if "leiden" in adata.obs.columns else "_batch")
    return sc_metrics.scib_score(
        knn_idx, knn_dist,
        batch_labels=adata.obs["_batch"].to_numpy(),
        label_labels=adata.obs[label_key].astype(str).to_numpy(),
    )

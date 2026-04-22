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
import scipy.sparse as sp

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


INTEGRATION_METHODS = ("none", "bbknn", "harmony")
"""Supported per-route integration methods. ``"none"`` = sc.pp.neighbors on
X_pca (batch-effect baseline). ``"bbknn"`` = scatlas.ext.bbknn (graph-level
batch correction). ``"harmony"`` = scatlas.ext.harmony → sc.pp.neighbors on
X_pca_harmony (embedding-level batch correction)."""


@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration.

    Each ``integration`` route runs an independent downstream chain
    (kNN → UMAP → scIB → Leiden → recall). When ``integration="all"``,
    every method in ``INTEGRATION_METHODS`` runs; results are stored under
    per-method keys (``X_umap_<method>``, ``leiden_<method>``, ...).
    """

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

    # --- HVG (03) — required, gates scale + PCA inputs
    hvg_n_top_genes: int = 2000
    hvg_flavor: str = "seurat_v3"         # VST on counts; see scatlas.pp.highly_variable_genes
    hvg_batch_aware: bool = True          # use batch_key to select HVGs per-batch, intersect

    # --- scale (04) — required, zeroes mean + clips z to [-10, 10]
    scale_max_value: float = 10.0
    scale_zero_center: bool = True

    # --- PCA (05)
    pca_n_comps: int | str = "auto"       # "auto" → Gavish-Donoho
    pca_n_power_iter: int = 7
    pca_random_state: int = 0

    # --- Integration route (06) ---------------------------------------------
    # "none"/"bbknn"/"harmony": run that single route.
    # "all": run every method in INTEGRATION_METHODS and produce comparison.
    integration: str = "bbknn"

    # BBKNN params (only used if integration in {"bbknn", "all"})
    neighbors_within_batch: int = 3
    bbknn_backend: str = "auto"           # "brute" / "hnsw" / "auto"

    # Harmony 2 params (only used if integration in {"harmony", "all"}).
    # Defaults diverge from R RunHarmony() in two places based on the
    # 157k-epithelia ablation (ablate_harmony.py):
    #   * theta=4 (not R's 2) — at theta=2 Harmony stalls at iter 2 with
    #     iLISI=0.086; theta=4 keeps iterating to iter 9, iLISI=0.174.
    #     cLISI trades down 0.91→0.83 but overall scIB mean is higher.
    #   * max_iter=20 (not R's 10) — gives theta=4 enough runway. No cost
    #     at theta=2 since convergence happens at iter 2 anyway.
    # If you need R-parity for a specific dataset, set theta=2, max_iter=10.
    harmony_n_clusters: int | None = None   # None → min(round(N/30), 100)
    harmony_max_iter: int = 20
    harmony_max_iter_cluster: int = 20
    harmony_theta: float = 4.0              # batch-diversity weight
    harmony_sigma: float = 0.1              # soft-cluster temperature
    harmony_lambda: float | None = 1.0      # None → dynamic α·E[k,b]
    harmony_alpha: float = 0.2              # dynamic-lambda scaling
    harmony_epsilon_cluster: float = 1e-3
    harmony_epsilon_harmony: float = 1e-2
    harmony_block_size: float = 0.05
    harmony_seed: int = 0

    # Plain kNN params (used for "none" + harmony's post-correction kNN).
    # Defaults match Seurat/SCOP scRNA convention (cosine, k=30) — gives
    # cleaner trajectory UMAPs than scanpy's euclidean/k=15 default.
    knn_n_neighbors: int = 30
    knn_metric: str = "cosine"

    # --- UMAP (07)
    run_umap: bool = True
    umap_min_dist: float = 0.5
    umap_spread: float = 1.0
    umap_n_epochs: int = 200
    # init="pca" matches Seurat/SCOP's Seurat::RunUMAP layout and on the
    # current stack (50 PCs + cosine k=30) avoids the old banding bug.
    # Set "spectral" if a specific dataset still collapses with "pca".
    umap_init: str = "pca"
    umap_random_state: int = 0

    # --- scIB metrics (08) — per-route
    run_metrics: bool = True
    label_key: str | None = None          # ground-truth cell-type; None → use per-route Leiden
    # silhouette is O(N²) in sklearn — at 157k it eats 7+ min per route.
    # Leave on for small/mid benchmarks; set False for atlas-scale runs.
    compute_silhouette: bool = True
    # Cluster-homogeneity metrics (ROGUE + SCCAF) — require Leiden first and
    # raw counts in layers['counts']. ROGUE scales with n_clusters × n_samples;
    # SCCAF is O(N·n_clusters) via sklearn LR. Both affordable at 157k if
    # silhouette is off.
    compute_homogeneity: bool = True
    # Optional auto-generated side-by-side comparison (needs >1 route).
    write_comparison_plot: str | None = None   # path to output PNG

    # --- Leiden (09) — per-route, auto-resolution
    # v1 defaults target MAJOR LINEAGE level (epithelia/immune/stromal/...),
    # not fine subtypes. Subclustering per lineage is a separate pass on a
    # subset adata with tuned target_n — see README.
    run_leiden: bool = True
    leiden_resolutions: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5])
    leiden_target_n: tuple[int, int] = (3, 10)    # pick smallest res giving k in [3, 10]
    leiden_n_iterations: int = 2

    # Resolution optimizer — "graph_silhouette" (default, data-driven via
    # graph-silhouette on subsamples) or "target_n" (legacy heuristic).
    resolution_optimizer: str = "graph_silhouette"
    silhouette_n_subsample: int = 1000
    silhouette_n_iter: int = 100
    silhouette_stratify: bool = True  # stratify by first baseline leiden res

    # --- recall (10) — per-route, auto-resolution via scvalidate
    # TEMPORARY (2026-04-22): recall reverted to opt-in default-off while the
    # atlas-scale performance is reworked. scvalidate's recall path is still
    # callable as a library; the pipeline just doesn't invoke it by default.
    # v1 spec intended recall mandatory; see docs/superpowers/specs/...
    run_recall: bool = False
    recall_resolution_start: float = 0.8
    recall_fdr: float = 0.05
    recall_max_iterations: int = 20
    recall_scratch_dir: str | None = None  # None → tempfile default

    def integration_methods(self) -> tuple[str, ...]:
        """Expand ``integration`` to the concrete list of routes to run."""
        if self.integration == "all":
            return INTEGRATION_METHODS
        if self.integration not in INTEGRATION_METHODS:
            raise ValueError(
                f"integration={self.integration!r} must be one of "
                f"{('all',) + INTEGRATION_METHODS}"
            )
        return (self.integration,)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _banner(s: str) -> None:
    print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")


def _step(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    dt = t1 - t0
    # show ms for sub-second steps so kNN / fuzzy don't appear as '0.0s'
    fmt = f"{dt * 1000:.0f}ms" if dt < 1.0 else f"{dt:.1f}s"
    print(f"  [{label}] {fmt}")
    return t1


def run_pipeline(
    input_h5ad: str | None = None,
    *, adata_in=None,
    **kwargs,
) -> Any:
    """Run the full pipeline. Returns the final AnnData.

    Either pass ``input_h5ad`` (+ keyword overrides) to read from disk, or
    pass ``adata_in=<AnnData>`` to run on an already-loaded object (useful
    for tests, rda-loaded data, or notebook workflows).
    """
    if adata_in is not None:
        cfg = PipelineConfig(input_h5ad=kwargs.pop("input_h5ad", "<in-memory>"), **kwargs)
        return run_from_config(cfg, adata_in=adata_in)
    if input_h5ad is not None:
        cfg = PipelineConfig(input_h5ad=input_h5ad, **kwargs)
    else:
        cfg = PipelineConfig(**kwargs)
    return run_from_config(cfg)


def run_from_config(cfg: PipelineConfig, *, adata_in=None) -> Any:
    import anndata as ad

    _banner(f"scatlas_pipeline — {cfg.input_h5ad}")
    pipeline_t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # --- 01 load + QC -------------------------------------------------------
    t0 = time.perf_counter()
    adata = adata_in if adata_in is not None else ad.read_h5ad(cfg.input_h5ad)
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

    # Preserve raw counts for recall (which needs integer counts,
    # not log1p) and for seurat_v3 HVG (fits VST on counts).
    adata.layers["counts"] = adata.X.copy()

    # --- 02 lognorm ---------------------------------------------------------
    t0 = time.perf_counter()
    adata.X = _lognorm(adata.X, cfg.target_sum)
    timings["lognorm"] = _step("02 lognorm", t0) - t0

    # --- 03 HVG -------------------------------------------------------------
    t0 = time.perf_counter()
    from scatlas import pp

    hvg_batch_key = "_batch" if cfg.hvg_batch_aware and len(uniq) > 1 else None
    hvg_layer = "counts" if cfg.hvg_flavor == "seurat_v3" else None
    pp.highly_variable_genes(
        adata,
        n_top_genes=cfg.hvg_n_top_genes,
        flavor=cfg.hvg_flavor,
        batch_key=hvg_batch_key,
        layer=hvg_layer,
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"         HVG ({cfg.hvg_flavor}, batch_aware={hvg_batch_key is not None}): "
          f"{n_hvg} / {adata.n_vars}")
    timings["hvg"] = _step("03 hvg", t0) - t0

    # --- 04 scale + 05 PCA on a disposable HVG-subsetted view --------------
    # Keep the main `adata` full-gene so layers['counts'] stays intact for
    # recall (needs raw counts) and downstream gene-level outputs.
    t0 = time.perf_counter()
    ad_hvg = adata[:, adata.var["highly_variable"]].copy()
    pp.scale(
        ad_hvg,
        max_value=cfg.scale_max_value,
        zero_center=cfg.scale_zero_center,
    )
    timings["scale"] = _step("04 scale", t0) - t0

    t0 = time.perf_counter()
    pca_out = pp.pca(
        ad_hvg,
        n_comps=cfg.pca_n_comps,
        n_power_iter=cfg.pca_n_power_iter,
        random_state=cfg.pca_random_state,
    )
    # Copy X_pca back to the main AnnData (same cell order, same n_obs).
    adata.obsm["X_pca"] = ad_hvg.obsm["X_pca"]
    adata.uns["pca"] = ad_hvg.uns.get("pca", {})
    del ad_hvg
    print(f"         X_pca {adata.obsm['X_pca'].shape}", end="")
    if pca_out.get("auto") is not None:
        info = pca_out["auto"]
        print(f"   GD={info['n_comps_gavish_donoho']} elbow={info['n_comps_elbow']}")
    else:
        print()
    timings["pca"] = _step("05 pca", t0) - t0

    # --- 06-10 per-integration-route ---------------------------------------
    n_batches = len(uniq)
    methods = cfg.integration_methods()
    # BBKNN + Harmony are batch-correction methods — meaningless on 1 batch
    # (BBKNN degenerates to k=neighbors_within_batch kNN; Harmony errors).
    # Drop them automatically and warn; keep "none" so the pipeline still
    # produces a valid output.
    if n_batches < 2:
        filtered = tuple(m for m in methods if m == "none")
        dropped = [m for m in methods if m != "none"]
        if dropped:
            print(f"\n[routes] only {n_batches} batch — skipping "
                  f"{dropped} (batch correction requires ≥ 2 batches)")
        if not filtered:
            # User asked only for bbknn/harmony but there's 1 batch; force-add none.
            filtered = ("none",)
            print("[routes] forcing integration='none' for 1-batch run")
        methods = filtered
    print(f"\n[routes] running integration methods: {list(methods)}")
    route_timings: dict[str, dict[str, float]] = {m: {} for m in methods}
    # Stash per-route kNN + embedding so Phase-2 can consume them without
    # recomputing.
    route_artifacts: dict[str, dict] = {}

    # ---------- Phase 1: fast path — integration + UMAP for all routes -----
    # Separating the fast path lets the user eyeball the comparison plot
    # long before the slow O(N²) scIB silhouettes finish.
    _banner("Phase 1: integration + UMAP (fast path)")
    for method in methods:
        print(f"\n── route: {method} (phase 1) ──")
        knn, conn, embed = _phase1_integration_umap(adata, method, cfg, route_timings[method])
        route_artifacts[method] = {"knn": knn, "conn": conn, "embed": embed}

    # ---------- Phase 1.5: write UMAP comparison before scIB starts --------
    if cfg.write_comparison_plot and len(methods) > 1:
        out_path = Path(cfg.write_comparison_plot)
        plot_path = compare_integration_plot(
            adata, out_path,
            label_key=cfg.label_key if (cfg.label_key and cfg.label_key in adata.obs.columns) else "_batch",
        )
        print(f"\n[comparison-plot] (pre-scIB) → {plot_path}")

    # ---------- Phase 2: slow path — scIB + Leiden + recall ---------------
    _banner("Phase 2: scIB + Leiden + recall (slow path)")
    for method in methods:
        print(f"\n── route: {method} (phase 2) ──")
        arts = route_artifacts[method]
        _phase2_metrics_cluster(
            adata, method, cfg, route_timings[method],
            knn=arts["knn"], embed=arts["embed"], conn=arts["conn"],
        )

    # --- all-mode: assemble scIB comparison table + heatmap ----------------
    if len(methods) > 1:
        _banner("scIB comparison across integration methods")
        scib_table = _scib_comparison_table(adata, methods)
        adata.uns["scib_comparison"] = scib_table
        for row in scib_table:
            print("  " + "  ".join(f"{k}={row[k]}" for k in row))

        # Heatmap sibling to write_comparison_plot — same stem, _scib.png suffix.
        if cfg.write_comparison_plot:
            src = Path(cfg.write_comparison_plot)
            heat = src.with_name(f"{src.stem}_scib.png")
            try:
                compare_scib_heatmap(adata, heat, methods=methods)
                print(f"\n[scib-heatmap] → {heat}")
            except Exception as e:
                print(f"[scib-heatmap] failed: {type(e).__name__}: {e}")

            # Per-cluster ROGUE bar plot — only if homogeneity ran.
            have_rogue = any(
                f"rogue_per_cluster_{m}" in adata.uns for m in methods
            )
            if have_rogue:
                rog = src.with_name(f"{src.stem}_rogue.png")
                try:
                    compare_rogue_per_cluster(adata, rog, methods=methods)
                    print(f"[rogue-per-cluster] → {rog}")
                except Exception as e:
                    print(f"[rogue-per-cluster] failed: {type(e).__name__}: {e}")

    # Flatten route timings into the top-level timings dict for display.
    for m, steps in route_timings.items():
        for k, v in steps.items():
            timings[f"{m}/{k}"] = v

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


def _phase1_integration_umap(
    adata, method: str, cfg: PipelineConfig, route_t: dict[str, float],
) -> tuple[dict, Any, np.ndarray]:
    """Phase 1 of a route — the fast, visible path.

    Runs integration (kNN + embedding) then UMAP. Returns
    ``(knn_dict, conn_csr, embed_for_scib)`` so Phase 2 can compute scIB
    without redoing kNN.
    """
    from scatlas import ext as sc_ext

    # --- Build kNN + embedding for this route (07 integration) -------------
    t0 = time.perf_counter()
    if method == "bbknn":
        # BBKNN: real batch labels, batch-balanced kNN on X_pca.
        knn, conn = _knn_and_fuzzy(
            adata.obsm["X_pca"], adata.obs["_batch"].to_numpy(),
            neighbors_within_batch=cfg.neighbors_within_batch,
            backend=cfg.bbknn_backend,
            metric=cfg.knn_metric,
        )
        embed_for_scib = adata.obsm["X_pca"]
        adata.obsm[f"X_pca_{method}"] = embed_for_scib
    elif method == "harmony":
        # Harmony: correct PCA first, then plain kNN on corrected embedding.
        sc_ext.harmony(
            adata,
            batch_key="_batch", use_rep="X_pca",
            n_clusters=cfg.harmony_n_clusters,
            theta=cfg.harmony_theta,
            sigma=cfg.harmony_sigma,
            lambda_=cfg.harmony_lambda,
            alpha=cfg.harmony_alpha,
            max_iter=cfg.harmony_max_iter,
            max_iter_cluster=cfg.harmony_max_iter_cluster,
            epsilon_cluster=cfg.harmony_epsilon_cluster,
            epsilon_harmony=cfg.harmony_epsilon_harmony,
            block_size=cfg.harmony_block_size,
            seed=cfg.harmony_seed,
        )
        # Log whether Harmony actually converged — hitting max_iter means
        # iLISI / kBET will likely underperform BBKNN even with correct
        # downstream params.
        h_info = adata.uns.get("harmony", {})
        converged = h_info.get("converged_at_iter", None)
        mode = (h_info.get("params", {}) or {}).get("lambda_mode", "?")
        if converged is not None:
            status = (f"converged at iter {converged}"
                      if converged < cfg.harmony_max_iter
                      else f"did NOT converge (ran full max_iter={cfg.harmony_max_iter})")
            print(f"         [harmony] {status}  (lambda={mode}, "
                  f"theta={cfg.harmony_theta}, sigma={cfg.harmony_sigma})")
        embed_for_scib = adata.obsm["X_pca_harmony"]
        adata.obsm[f"X_pca_{method}"] = embed_for_scib
        # Dummy single-batch labels → plain kNN on corrected embedding.
        dummy_batch = np.zeros(adata.n_obs, dtype=np.int32)
        knn, conn = _knn_and_fuzzy(
            embed_for_scib, dummy_batch,
            neighbors_within_batch=cfg.knn_n_neighbors,
            backend=cfg.bbknn_backend,
            metric=cfg.knn_metric,
        )
    elif method == "none":
        # Baseline: plain kNN on uncorrected X_pca (shows batch effect).
        dummy_batch = np.zeros(adata.n_obs, dtype=np.int32)
        knn, conn = _knn_and_fuzzy(
            adata.obsm["X_pca"], dummy_batch,
            neighbors_within_batch=cfg.knn_n_neighbors,
            backend=cfg.bbknn_backend,
            metric=cfg.knn_metric,
        )
        embed_for_scib = adata.obsm["X_pca"]
        adata.obsm[f"X_pca_{method}"] = embed_for_scib
    else:
        raise ValueError(f"unknown integration method: {method!r}")

    adata.obsp[f"{method}_connectivities"] = conn
    adata.uns[f"{method}_knn"] = {
        "indices": knn["indices"], "distances": knn["distances"],
    }
    MAX = np.iinfo(np.uint32).max
    valid_edges = int((knn["indices"] != MAX).sum())
    print(f"         [{method}] kNN: shape={knn['indices'].shape} "
          f"backend={knn['backend_used']} valid_edges={valid_edges}")
    print(f"         [{method}] fuzzy CSR: nnz={conn.nnz}")
    route_t["integration"] = _step(f"07 {method}", t0) - t0

    # --- 08 UMAP on the route's connectivity graph -------------------------
    if cfg.run_umap:
        t0 = time.perf_counter()
        _run_umap_for_route(adata, method, conn, cfg)
        route_t["umap"] = _step(f"08 {method}/umap", t0) - t0

    return knn, conn, embed_for_scib


def _phase2_metrics_cluster(
    adata, method: str, cfg: PipelineConfig, route_t: dict[str, float],
    *, knn: dict, embed: np.ndarray, conn: Any,
) -> None:
    """Phase 2 — slow path (scIB silhouettes O(N²), Leiden sweep, ROGUE,
    SCCAF, recall).

    Metric order aligns with Zhang-lab Cross-tissue fibroblast atlas (Cancer
    Cell 2024): bio-conservation + batch-removal (scIB core) first, then
    after Leiden the cluster-homogeneity metrics (ROGUE + SCCAF).
    """
    # --- 09 scIB core (bio + batch, not cluster-homogeneity) ---------------
    if cfg.run_metrics:
        t0 = time.perf_counter()
        scib = _compute_scib_for_route(adata, method, knn, embed, cfg)
        adata.uns[f"scib_{method}"] = scib
        for k, v in scib.items():
            if isinstance(v, (int, float)):
                print(f"             {k:22s} = {v:.3f}")
        route_t["metrics"] = _step(f"09 {method}/metrics", t0) - t0

    # --- 10 Leiden (auto resolution) on the route's connectivities ---------
    if cfg.run_leiden:
        t0 = time.perf_counter()
        labels, chosen_res = _leiden_auto_resolution(adata, method, conn, cfg)
        adata.obs[f"leiden_{method}"] = labels
        adata.uns[f"leiden_{method}_resolution"] = chosen_res
        print(f"         [{method}] picked r={chosen_res} → "
              f"{len(np.unique(labels))} clusters")
        route_t["leiden"] = _step(f"10 {method}/leiden", t0) - t0

    # --- 10.5 cluster-homogeneity (ROGUE + SCCAF) after Leiden -------------
    if cfg.run_metrics and cfg.run_leiden and cfg.compute_homogeneity:
        _compute_homogeneity_for_route(adata, method, embed, cfg, route_t)

    # --- 11 recall (opt-in, default off — atlas-scale perf rework pending) -
    if cfg.run_recall:
        t0 = time.perf_counter()
        _run_recall_for_route(adata, method, cfg)
        route_t["recall"] = _step(f"11 {method}/recall", t0) - t0


def _compute_homogeneity_for_route(
    adata, method: str, embed: np.ndarray, cfg: PipelineConfig,
    route_t: dict[str, float],
) -> None:
    """ROGUE (scvalidate Rust kernel) + SCCAF-equivalent (sklearn LR CV).

    ROGUE per-cluster is stored at ``uns['rogue_per_cluster_<method>']``
    (user feedback: must surface per-cluster ROGUE, not just a verdict).
    ROGUE mean + SCCAF accuracy merge into ``uns['scib_<method>']`` as
    homogeneity scores for the heatmap.
    """
    from scatlas import metrics as sc_metrics

    cluster_labels = adata.obs[f"leiden_{method}"].astype(str).to_numpy()
    sample_labels = adata.obs.get("_batch", None)
    if sample_labels is not None:
        sample_labels = sample_labels.astype(str).to_numpy()

    # --- ROGUE (Rust via scvalidate_rust.entropy_table + calculate_rogue) --
    t0 = time.perf_counter()
    try:
        # scvalidate expects raw counts (genes × cells). Preserved in layers.
        counts_gxc = (adata.layers["counts"]
                      if sp.issparse(adata.layers["counts"])
                      else sp.csr_matrix(adata.layers["counts"])).T
        rogue = sc_metrics.rogue_mean(
            counts_gxc, cluster_labels, sample_labels,
            platform="UMI",
            gene_names=adata.var_names.tolist() if hasattr(adata, "var_names") else None,
        )
        adata.uns[f"rogue_per_cluster_{method}"] = rogue["per_cluster"]
        adata.uns[f"scib_{method}"]["rogue_mean"] = rogue["mean"]
        adata.uns[f"scib_{method}"]["rogue_median"] = rogue["median"]
        print(f"             {'rogue_mean':22s} = {rogue['mean']:.3f}  "
              f"(per-cluster n={rogue['n_clusters_scored']})")
    except Exception as e:
        print(f"         [{method}] ROGUE failed: {type(e).__name__}: {e}")
    route_t["rogue"] = _step(f"10a {method}/rogue", t0) - t0

    # --- SCCAF-equivalent (sklearn logistic-regression CV accuracy) --------
    t0 = time.perf_counter()
    try:
        sccaf_acc = sc_metrics.sccaf_accuracy(embed, cluster_labels)
        adata.uns[f"scib_{method}"]["sccaf"] = sccaf_acc
        print(f"             {'sccaf':22s} = {sccaf_acc:.3f}")
    except Exception as e:
        print(f"         [{method}] SCCAF failed: {type(e).__name__}: {e}")
    route_t["sccaf"] = _step(f"10b {method}/sccaf", t0) - t0


def _knn_and_fuzzy(
    embedding: np.ndarray, batch_codes: np.ndarray,
    neighbors_within_batch: int, backend: str,
    metric: str = "cosine",
) -> tuple[dict, Any]:
    """Rust kNN (batch-balanced if multi-batch, plain if all-zero) + fuzzy
    simplicial set → (knn dict, CSR connectivities).

    ``metric="cosine"`` L2-normalizes the embedding before passing to the
    euclidean-only BBKNN kernel. On unit-norm vectors the top-k nearest
    neighbors by euclidean distance are **exactly** the top-k nearest by
    cosine — the standard trick scanpy uses internally. This matters for
    scRNA trajectory data where cosine gives much cleaner UMAPs than
    euclidean (insensitive to cell-level magnitude).
    """
    from scatlas import ext as sc_ext
    from scatlas.ext import _fuzzy_connectivities

    if metric == "cosine":
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embedding = (embedding / norms).astype(np.float32, copy=False)
    elif metric != "euclidean":
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")

    knn = sc_ext.bbknn_kneighbors(
        embedding, batch_codes,
        neighbors_within_batch=int(neighbors_within_batch),
        backend=backend,
    )
    conn = _fuzzy_connectivities(
        knn["indices"], knn["distances"], embedding.shape[0],
    )
    return knn, conn


def _run_umap_for_route(adata, method: str, conn, cfg: PipelineConfig) -> None:
    """Run scatlas UMAP on this route's CSR, writing ``obsm[X_umap_<method>]``.

    We bypass the scatlas.tl.umap convenience wrapper (which reads a fixed
    obsp key) and drive the Rust kernel directly so each route can own its
    own graph.
    """
    from scatlas import tl
    from scatlas._scatlas_native import tl as _rust_tl

    key = f"{method}_connectivities"
    # Temporarily expose under the wrapper's expected key and call tl.umap
    # so init='spectral' / _pick_init machinery runs uniformly.
    adata.obsp[key] = conn
    tl.umap(
        adata,
        neighbors_key=method,
        min_dist=cfg.umap_min_dist,
        spread=cfg.umap_spread,
        n_epochs=cfg.umap_n_epochs,
        init=cfg.umap_init,
        random_state=cfg.umap_random_state,
    )
    adata.obsm[f"X_umap_{method}"] = adata.obsm.pop("X_umap")
    adata.uns[f"umap_{method}"] = adata.uns.pop("umap")
    _ = _rust_tl  # keep import for potential future direct-kernel call


def _compute_scib_for_route(
    adata, method: str, knn: dict, embedding: np.ndarray, cfg: PipelineConfig,
) -> dict[str, Any]:
    """scatlas Rust scIB on this route's kNN + batch/label pair."""
    from scatlas import metrics as sc_metrics

    MAX = np.iinfo(np.uint32).max
    indices_u32 = knn["indices"]
    sentinel_mask = indices_u32 == MAX
    knn_idx = indices_u32.astype(np.int32)
    knn_dist = knn["distances"].astype(np.float32).copy()
    if sentinel_mask.any():
        row_idx = np.broadcast_to(
            np.arange(knn_idx.shape[0], dtype=np.int32)[:, None], knn_idx.shape
        )
        knn_idx = np.where(sentinel_mask, row_idx, knn_idx)
        knn_dist[sentinel_mask] = 0.0

    # Prefer ground truth if provided; else fall back to this route's Leiden.
    if cfg.label_key and cfg.label_key in adata.obs.columns:
        label_src = cfg.label_key
    elif f"leiden_{method}" in adata.obs.columns:
        label_src = f"leiden_{method}"
    else:
        label_src = "_batch"
    label_arr = adata.obs[label_src].astype(str).to_numpy()

    embed_for_scib = None
    if cfg.compute_silhouette:
        embed_for_scib = np.ascontiguousarray(embedding, dtype=np.float32)
    return sc_metrics.scib_score(
        knn_idx, knn_dist,
        batch_labels=adata.obs["_batch"].to_numpy(),
        label_labels=label_arr,
        embedding=embed_for_scib,
    )


def _leiden_auto_resolution(
    adata, method: str, conn, cfg: PipelineConfig,
) -> tuple[np.ndarray, float]:
    """Choose a Leiden resolution using the configured optimizer.

    Two modes:
      - "target_n" (legacy): smallest res giving n_clusters in leiden_target_n.
      - "graph_silhouette" (default v1): run all resolutions, pick the one
        with highest mean graph-silhouette on subsamples; clip eligibility
        to n_clusters ∈ leiden_target_n as a sanity bound.
    """
    import scanpy as sc
    from scatlas_pipeline.silhouette import (
        optimize_resolution_graph_silhouette, pick_best_resolution,
    )

    # Make scanpy leiden read the route's connectivities
    adata.obsp["connectivities"] = conn
    adata.uns["neighbors"] = {
        "params": {"method": "umap"},
        "connectivities_key": "connectivities",
    }

    if cfg.resolution_optimizer == "target_n":
        # ── legacy target_n path ───────────────────────────────────────
        chosen: tuple[float, np.ndarray] | None = None
        for r in cfg.leiden_resolutions:
            key = f"_leiden_{method}_r{r}"
            sc.tl.leiden(
                adata, resolution=r, key_added=key,
                flavor="igraph", directed=False,
                n_iterations=cfg.leiden_n_iterations, random_state=0,
            )
            n = adata.obs[key].nunique()
            in_range = cfg.leiden_target_n[0] <= n <= cfg.leiden_target_n[1]
            print(f"         [{method}] leiden r={r} → {n} clusters"
                  f"{' ✓' if in_range else ''}")
            if in_range and chosen is None:
                chosen = (r, adata.obs[key].to_numpy())
        if chosen is None:
            r = cfg.leiden_resolutions[len(cfg.leiden_resolutions) // 2]
            print(f"         [{method}] [fallback] no resolution in target; r={r}")
            chosen = (r, adata.obs[f"_leiden_{method}_r{r}"].to_numpy())
        return chosen[1], chosen[0]

    if cfg.resolution_optimizer != "graph_silhouette":
        raise ValueError(
            f"resolution_optimizer={cfg.resolution_optimizer!r} — "
            f"must be 'target_n' or 'graph_silhouette'"
        )

    # ── graph_silhouette path ────────────────────────────────────────────
    stratify_key = None
    if cfg.silhouette_stratify:
        # One quick coarse leiden to use as stratification key — use the
        # smallest resolution in the sweep list.
        r0 = min(cfg.leiden_resolutions)
        k0 = f"_leiden_{method}_r{r0}_strata"
        sc.tl.leiden(
            adata, resolution=r0, key_added=k0,
            flavor="igraph", directed=False,
            n_iterations=cfg.leiden_n_iterations, random_state=0,
        )
        stratify_key = k0

    curve = optimize_resolution_graph_silhouette(
        adata,
        method=method,
        conn=conn,
        neighbors_key=None,  # use the generic obsp["connectivities"]
        resolutions=cfg.leiden_resolutions,
        n_subsample=cfg.silhouette_n_subsample,
        n_iter=cfg.silhouette_n_iter,
        stratify_key=stratify_key,
        seed=0,
        leiden_flavor="igraph",
        leiden_n_iterations=cfg.leiden_n_iterations,
        verbose=True,
    )
    adata.uns[f"silhouette_curve_{method}"] = {
        "resolution":      curve["resolution"].tolist(),
        "mean_silhouette": curve["mean_silhouette"].tolist(),
        "sd_silhouette":   curve["sd_silhouette"].tolist(),
        "n_clusters":      curve["n_clusters"].tolist(),
    }

    # Pick best within target_n bounds if they're meaningful; else unclipped.
    k_lo, k_hi = cfg.leiden_target_n
    try:
        best_r = pick_best_resolution(curve, k_lo=k_lo, k_hi=k_hi)
    except ValueError:
        # No resolution produces k in [k_lo, k_hi] — fall back to unclipped max
        best_r = pick_best_resolution(curve)
        print(f"         [{method}] [silhouette] no res in k∈[{k_lo},{k_hi}]; "
              f"unclipped best r={best_r}")

    best_labels_key = f"leiden_{method}_r{best_r:.2f}"
    if best_labels_key not in adata.obs.columns:
        # safety: the optimizer should have written this; regenerate if missing
        sc.tl.leiden(
            adata, resolution=best_r, key_added=best_labels_key,
            flavor="igraph", directed=False,
            n_iterations=cfg.leiden_n_iterations, random_state=0,
        )
    labels = adata.obs[best_labels_key].to_numpy()

    print(f"         [{method}] [silhouette] picked r={best_r:.2f} by "
          f"mean_silhouette (curve stored in uns)")
    return labels, best_r


def _run_recall_for_route(adata, method: str, cfg: PipelineConfig) -> None:
    """scvalidate recall on raw counts with oom backend for >=30k cells.

    Also emits RecallComparisonReport comparing:
      - baseline Leiden k (from step 10, selected by cfg.leiden_target_n)
      - recall-calibrated k (this step)
    stored in adata.uns[f"recall_{method}_comparison"].
    """
    import time
    try:
        from scvalidate.recall_py import (
            find_clusters_recall, build_comparison_report,
        )
    except ImportError:
        print(
            f"         [{method}] [recall] scvalidate not installed — skipping "
            f"(temporary: recall is opt-in while atlas-scale perf is reworked)"
        )
        return

    # Raw counts in layers["counts"] (preserved before lognorm)
    X = adata.layers["counts"]
    counts_gxc = (X if sp.issparse(X) else sp.csr_matrix(X)).T

    # Baseline labels from step 10 (Leiden + target_n)
    baseline_key = f"leiden_{method}"
    if baseline_key not in adata.obs.columns:
        print(f"         [{method}] [recall] no baseline leiden — skipping")
        return
    labels_baseline = adata.obs[baseline_key].astype(int).to_numpy()
    # Find the selected baseline resolution (stored by step 10)
    res_baseline = float(adata.uns.get(f"leiden_{method}_resolution", 0.8))

    # Anchor recall at baseline's auto-selected resolution rather than the
    # Seurat-inherited 0.8 default. target_n is scale-adaptive (157k lands
    # around res=0.3-0.5 for k∈[8,30]); starting recall there means recall
    # verifies the baseline choice and only steps down if a pair is
    # knockoff-indistinguishable. On k=40 at res=0.8 baseline, naive start
    # would walk 4-6 iterations × O(k²) wilcoxon; from res_baseline recall
    # typically converges in 1-2 iter.
    t0 = time.perf_counter()
    result = find_clusters_recall(
        counts_gxc,
        resolution_start=res_baseline,
        fdr=cfg.recall_fdr,
        max_iterations=cfg.recall_max_iterations,
        seed=0,
        backend="auto",
        scratch_dir=cfg.recall_scratch_dir,
    )
    wall = time.perf_counter() - t0

    adata.obs[f"recall_{method}"] = result.labels.astype(str)
    adata.uns[f"recall_{method}_resolution"] = result.resolution
    adata.uns[f"recall_{method}_iterations"] = result.n_iterations

    # Comparison report
    report = build_comparison_report(
        labels_baseline=labels_baseline,
        labels_recall=result.labels,
        resolution_baseline=res_baseline,
        resolution_recall=result.resolution,
        recall_converged=result.converged,
        k_trajectory=result.k_trajectory,
        recall_wall_time_s=wall,
    )
    adata.uns[f"recall_{method}_comparison"] = report.to_dict()

    print(
        f"         [{method}] recall: k_baseline={report.k_baseline} → "
        f"k_recall={report.k_recall} (ΔK={report.delta_k}), "
        f"ARI={report.ari_baseline_vs_recall:.3f}, "
        f"converged={report.recall_converged}, "
        f"wall={wall:.1f}s"
    )


def compare_integration_plot(
    adata, out_path: str | Path,
    *, label_key: str | None = None, methods: tuple[str, ...] | None = None,
    point_size: float = 4.0, dpi: int = 150,
) -> Path:
    """Side-by-side UMAP grid for all integration routes present on the AnnData.

    Looks up ``obsm[f'X_umap_{method}']`` for each route. Colors points by
    ``label_key`` if given (ground truth) else by ``_batch`` (shows batch
    effect). Saves PNG and returns the path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if methods is None:
        methods = tuple(
            m for m in INTEGRATION_METHODS
            if f"X_umap_{m}" in adata.obsm
        )
    if not methods:
        raise ValueError("no per-route UMAP embeddings found on adata.obsm")

    color_key = label_key or "_batch"
    if color_key not in adata.obs.columns:
        raise ValueError(f"obs[{color_key!r}] not found — can't color")
    labels = adata.obs[color_key].astype(str).to_numpy()
    uniq = np.unique(labels)
    cmap = plt.colormaps["tab20"] if len(uniq) <= 20 else plt.colormaps["gist_ncar"]

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)
    for ax, method in zip(axes[0], methods):
        emb = adata.obsm[f"X_umap_{method}"]
        for i, lab in enumerate(uniq):
            mask = labels == lab
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                s=point_size, alpha=0.75,
                color=cmap(i / max(len(uniq) - 1, 1)),
                label=lab if len(uniq) <= 12 else None,
            )
        scib = adata.uns.get(f"scib_{method}", {})
        sub = " / ".join(
            f"{k[:5]}={scib[k]:.2f}"
            for k in ("ilisi", "clisi", "graph_connectivity", "mean")
            if k in scib and isinstance(scib[k], (int, float))
        )
        ax.set_title(f"{method}\n{sub}" if sub else method, fontsize=10)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        if len(uniq) <= 12:
            ax.legend(loc="best", fontsize=7, frameon=False)

    fig.suptitle(f"integration comparison — colored by {color_key}", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


SCIB_BATCH_METRICS = ("ilisi", "kbet_acceptance", "batch_silhouette")
SCIB_BIO_METRICS = ("clisi", "graph_connectivity", "label_silhouette", "isolated_label")
SCIB_HOMO_METRICS = ("rogue_mean", "sccaf")
SCIB_METRIC_DISPLAY = {
    "ilisi": "iLISI",
    "kbet_acceptance": "kBET",
    "batch_silhouette": "batch ASW",
    "clisi": "cLISI",
    "graph_connectivity": "graph conn",
    "label_silhouette": "label ASW",
    "isolated_label": "iso label",
    "rogue_mean": "ROGUE",
    "sccaf": "SCCAF",
}


def compare_scib_heatmap(
    adata, out_path: str | Path,
    *, methods: tuple[str, ...] | None = None,
    dpi: int = 150,
) -> Path:
    """scib-benchmark + Zhang-lab-style heatmap: methods × metrics, with
    three-category summary columns.

    Categories align with Zhang-lab Cross-tissue fibroblast atlas framework:
      * **Batch** = mean of iLISI / kBET / batch-silhouette  (batch removal)
      * **Bio**   = mean of cLISI / graph_connectivity / label-silhouette
                    / isolated-label-silhouette  (biology preserved)
      * **Homogeneity** = mean of ROGUE / SCCAF  (clusters are coherent)
      * **Overall** = 0.35·Batch + 0.45·Bio + 0.20·Homogeneity
                      (scib-benchmark weights bio 0.6/batch 0.4; here we
                       redistribute to make room for homogeneity)

    Cells missing a metric (e.g., silhouettes / ROGUE when disabled) are
    drawn gray and dropped from the category average for that row.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if methods is None:
        methods = tuple(
            m for m in INTEGRATION_METHODS if f"scib_{m}" in adata.uns
        )
    if not methods:
        raise ValueError("no scib_<method> entries in adata.uns")

    # Individual columns: Batch | Bio | Homogeneity.
    all_metrics = (list(SCIB_BATCH_METRICS) + list(SCIB_BIO_METRICS)
                   + list(SCIB_HOMO_METRICS))
    rows: list[list[float | None]] = []
    for m in methods:
        scib = adata.uns.get(f"scib_{m}", {})
        row = [float(scib[k]) if k in scib and isinstance(scib[k], (int, float)) else None
               for k in all_metrics]
        batch_vals = [row[i] for i, k in enumerate(all_metrics)
                      if k in SCIB_BATCH_METRICS and row[i] is not None]
        bio_vals = [row[i] for i, k in enumerate(all_metrics)
                    if k in SCIB_BIO_METRICS and row[i] is not None]
        homo_vals = [row[i] for i, k in enumerate(all_metrics)
                     if k in SCIB_HOMO_METRICS and row[i] is not None]
        batch_s = float(np.mean(batch_vals)) if batch_vals else None
        bio_s = float(np.mean(bio_vals)) if bio_vals else None
        homo_s = float(np.mean(homo_vals)) if homo_vals else None
        # Overall = 0.35 Batch + 0.45 Bio + 0.20 Homogeneity (renormalized
        # if any dimension is missing).
        parts = [(0.35, batch_s), (0.45, bio_s), (0.20, homo_s)]
        usable = [(w, v) for w, v in parts if v is not None]
        if usable:
            total_w = sum(w for w, _ in usable)
            overall = sum(w * v for w, v in usable) / total_w
        else:
            overall = None
        rows.append(row + [batch_s, bio_s, homo_s, overall])

    col_labels = [SCIB_METRIC_DISPLAY[k] for k in all_metrics] + [
        "Batch", "Bio", "Homo", "Overall"
    ]
    arr = np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)
    n_rows, n_cols = arr.shape

    # Figure sizing scales with row/col count.
    fig_w = max(7.5, 0.95 * n_cols)
    fig_h = max(2.2, 0.65 * n_rows + 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw heatmap — mask NaN to gray. Use viridis (higher = better, green).
    cmap = plt.colormaps["viridis"].with_extremes(bad="#cccccc")
    im = ax.imshow(
        np.ma.masked_invalid(arr),
        cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0,
    )

    # Visual separator: draw a vertical line between individual metrics
    # and the summary columns.
    n_individual = len(all_metrics)
    ax.axvline(n_individual - 0.5, color="white", lw=2.5)

    # Cell text annotations.
    for i in range(n_rows):
        for j in range(n_cols):
            v = arr[i, j]
            if np.isnan(v):
                txt = "—"
                color = "black"
            else:
                txt = f"{v:.2f}"
                # Contrast: low values get light text on dark viridis.
                color = "white" if v < 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(methods), fontsize=10)

    # Column group annotations — three dimensions + summary.
    nb = len(SCIB_BATCH_METRICS)
    nbi = len(SCIB_BIO_METRICS)
    nh = len(SCIB_HOMO_METRICS)
    ax.text(
        (nb - 1) / 2, -0.85, "batch mixing ↑",
        ha="center", va="bottom", fontsize=9, color="#444",
        transform=ax.transData,
    )
    ax.text(
        nb + (nbi - 1) / 2, -0.85, "bio conservation ↑",
        ha="center", va="bottom", fontsize=9, color="#444",
        transform=ax.transData,
    )
    ax.text(
        nb + nbi + (nh - 1) / 2, -0.85, "cluster homogeneity ↑",
        ha="center", va="bottom", fontsize=9, color="#444",
        transform=ax.transData,
    )
    ax.text(
        n_individual + 1.5, -0.85, "summary",
        ha="center", va="bottom", fontsize=9, color="#444",
        transform=ax.transData,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("score (higher = better)", fontsize=9)

    ax.set_title("scIB — integration method comparison", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compare_rogue_per_cluster(
    adata, out_path: str | Path,
    *, methods: tuple[str, ...] | None = None,
    top_n: int | None = None, dpi: int = 150,
) -> Path:
    """Per-cluster ROGUE bar plot — one panel per integration route.

    For each method, read ``adata.uns['rogue_per_cluster_<m>']`` and plot
    a bar chart sorted by ROGUE descending, colored by purity band
    (green ≥ 0.85, yellow 0.70-0.85, red < 0.70). Shows **which clusters
    need sub-clustering** — low-ROGUE clusters mix multiple cell types.

    User feedback: must surface per-cluster ROGUE visually, not just a
    single mean in the heatmap. This complements ``compare_scib_heatmap``.

    Parameters
    ----------
    top_n
        If set, plot only top-N + bottom-N clusters by ROGUE (useful when
        N_clusters > 50). Default None shows all clusters per method.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if methods is None:
        methods = tuple(
            m for m in INTEGRATION_METHODS
            if f"rogue_per_cluster_{m}" in adata.uns
        )
    if not methods:
        raise ValueError("no rogue_per_cluster_<method> entries in adata.uns")

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(max(5.0, 5.5 * n), 5), squeeze=False)
    for ax, m in zip(axes[0], methods):
        pc = adata.uns.get(f"rogue_per_cluster_{m}", {})
        if not pc:
            ax.text(0.5, 0.5, f"no ROGUE for {m}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        items = sorted(pc.items(), key=lambda kv: kv[1], reverse=True)
        if top_n is not None and len(items) > 2 * top_n:
            items = items[:top_n] + items[-top_n:]
        labels = [str(k) for k, _ in items]
        values = np.array([v for _, v in items], dtype=float)
        # Purity band coloring: green ≥ 0.85, yellow 0.70-0.85, red < 0.70.
        colors = np.where(
            values >= 0.85, "#2a9d8f",
            np.where(values >= 0.70, "#e9c46a", "#e76f51"),
        )
        bars = ax.bar(range(len(values)), values, color=colors)
        ax.axhline(0.85, color="#2a9d8f", linestyle="--", alpha=0.4, lw=1)
        ax.axhline(0.70, color="#e76f51", linestyle="--", alpha=0.4, lw=1)

        scib = adata.uns.get(f"scib_{m}", {})
        rogue_mean = scib.get("rogue_mean", None)
        title = m if rogue_mean is None else f"{m}  (mean={rogue_mean:.2f}, n={len(pc)})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("cluster (sorted by ROGUE)")
        ax.set_ylabel("ROGUE purity (↑)")
        ax.set_ylim(0, 1.05)
        if len(labels) <= 20:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        else:
            # Too many clusters — hide x-tick labels, just show index.
            ax.set_xticks([0, len(labels) // 2, len(labels) - 1])
            ax.set_xticklabels(
                [labels[0], labels[len(labels) // 2], labels[-1]],
                fontsize=7,
            )

    fig.suptitle(
        "per-cluster ROGUE — green ≥ 0.85 pure · yellow 0.70-0.85 · red < 0.70 mixed",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _scib_comparison_table(adata, methods: tuple[str, ...]) -> list[dict]:
    """Build a list-of-dicts comparison table for all routes' scIB scores.
    Written into ``adata.uns['scib_comparison']`` for downstream plotting /
    bench-driven regression checks."""
    rows: list[dict] = []
    for m in methods:
        key = f"scib_{m}"
        if key not in adata.uns:
            continue
        rec = adata.uns[key]
        row: dict[str, Any] = {"method": m}
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                row[k] = round(float(v), 4)
        rows.append(row)
    return rows



"""End-to-end pipeline driver. Ported from v1 ``scatlas_pipeline/pipeline.py``
at V2-P2 with all recall paths stripped."""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from .config import PipelineConfig, INTEGRATION_METHODS

warnings.filterwarnings("ignore")


def _banner(s: str) -> None:
    print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")


def _step(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    dt = t1 - t0
    fmt = f"{dt * 1000:.0f}ms" if dt < 1.0 else f"{dt:.1f}s"
    print(f"  [{label}] {fmt}")
    return t1


def run_pipeline(
    input_h5ad: str | None = None,
    *, adata_in=None,
    **kwargs,
) -> Any:
    """Run the full pipeline. Returns the final AnnData."""
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

    from . import preprocess as pp
    from . import pca as _pca
    from .io import qc_filter

    _banner(f"fast_auto_scrna — {cfg.input_h5ad}")
    pipeline_t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # --- 01 load + QC
    t0 = time.perf_counter()
    adata = adata_in if adata_in is not None else ad.read_h5ad(cfg.input_h5ad)
    print(f"[01 load]   raw: {adata.n_obs} × {adata.n_vars}")
    adata = qc_filter(adata, cfg)
    print(f"         post-QC: {adata.n_obs} × {adata.n_vars}")
    timings["load+qc"] = _step("01 load+qc", t0) - t0

    adata.obs["_batch"] = adata.obs[cfg.batch_key].astype(str)
    uniq, cnt = np.unique(adata.obs["_batch"], return_counts=True)
    print(f"         batches: {dict(zip(uniq.tolist(), cnt.tolist()))}")

    # Preserve raw counts for HVG (seurat_v3 fits VST on counts) and ROGUE.
    adata.layers["counts"] = adata.X.copy()

    # --- 02 lognorm
    t0 = time.perf_counter()
    adata.X = pp.lognorm(adata.X, cfg.target_sum)
    timings["lognorm"] = _step("02 lognorm", t0) - t0

    # --- 03 HVG
    t0 = time.perf_counter()
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

    # --- 04 scale + 05 PCA on HVG subset view (main adata stays full-gene)
    t0 = time.perf_counter()
    ad_hvg = adata[:, adata.var["highly_variable"]].copy()
    pp.scale(
        ad_hvg,
        max_value=cfg.scale_max_value,
        zero_center=cfg.scale_zero_center,
    )
    timings["scale"] = _step("04 scale", t0) - t0

    t0 = time.perf_counter()
    pca_out = _pca.pca(
        ad_hvg,
        n_comps=cfg.pca_n_comps,
        n_power_iter=cfg.pca_n_power_iter,
        random_state=cfg.pca_random_state,
    )
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

    # --- 06-10 per-route
    n_batches = len(uniq)
    methods = cfg.integration_methods()
    # BBKNN + Harmony on 1 batch is meaningless — drop them and keep "none".
    if n_batches < 2:
        filtered = tuple(m for m in methods if m == "none")
        dropped = [m for m in methods if m != "none"]
        if dropped:
            print(f"\n[routes] only {n_batches} batch — skipping "
                  f"{dropped} (batch correction requires ≥ 2 batches)")
        if not filtered:
            filtered = ("none",)
            print("[routes] forcing integration='none' for 1-batch run")
        methods = filtered
    print(f"\n[routes] running integration methods: {list(methods)}")
    route_timings: dict[str, dict[str, float]] = {m: {} for m in methods}
    route_artifacts: dict[str, dict] = {}

    # Phase 1 — integration + UMAP for every route
    _banner("Phase 1: integration + UMAP (fast path)")
    for method in methods:
        print(f"\n── route: {method} (phase 1) ──")
        knn, conn, embed = _phase1_integration_umap(
            adata, method, cfg, route_timings[method],
        )
        route_artifacts[method] = {"knn": knn, "conn": conn, "embed": embed}

    # Pre-scIB comparison UMAP plot
    if cfg.write_comparison_plot and len(methods) > 1:
        from .plotting import compare_integration_plot
        out_path = Path(cfg.write_comparison_plot)
        plot_path = compare_integration_plot(
            adata, out_path,
            label_key=cfg.label_key if (cfg.label_key and cfg.label_key in adata.obs.columns) else "_batch",
        )
        print(f"\n[comparison-plot] (pre-scIB) → {plot_path}")

    # Phase 2 — slow path: scIB + Leiden + homogeneity
    _banner("Phase 2: scIB + Leiden + homogeneity (slow path)")
    for method in methods:
        print(f"\n── route: {method} (phase 2) ──")
        arts = route_artifacts[method]
        _phase2_metrics_cluster(
            adata, method, cfg, route_timings[method],
            knn=arts["knn"], embed=arts["embed"], conn=arts["conn"],
        )

    # Per-route diagnostic plots
    if cfg.plot_dir:
        from .plotting import emit_route_plots
        _banner(f"Per-route plots → {cfg.plot_dir}")
        for method in methods:
            written = emit_route_plots(adata, method, cfg.plot_dir, cfg)
            if written:
                print(f"  [{method}] wrote {len(written)} plots:")
                for p in written:
                    print(f"    {p.name}")

    # all-mode: assemble scIB comparison table + heatmap + rogue bars
    if len(methods) > 1:
        from .plotting import (
            scib_comparison_table, compare_scib_heatmap,
            compare_rogue_per_cluster,
        )
        _banner("scIB comparison across integration methods")
        table = scib_comparison_table(adata, methods)
        adata.uns["scib_comparison"] = table
        for row in table:
            print("  " + "  ".join(f"{k}={row[k]}" for k in row))

        if cfg.write_comparison_plot:
            src = Path(cfg.write_comparison_plot)
            heat = src.with_name(f"{src.stem}_scib.png")
            try:
                compare_scib_heatmap(adata, heat, methods=methods)
                print(f"\n[scib-heatmap] → {heat}")
            except Exception as e:
                print(f"[scib-heatmap] failed: {type(e).__name__}: {e}")

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

    for m, steps in route_timings.items():
        for k, v in steps.items():
            timings[f"{m}/{k}"] = v

    total = time.perf_counter() - pipeline_t0
    adata.uns["fast_auto_scrna_timings"] = timings
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


def _phase1_integration_umap(
    adata, method: str, cfg: PipelineConfig, route_t: dict[str, float],
) -> tuple[dict, Any, np.ndarray]:
    """Phase 1 of a route — integration (kNN + embedding) then UMAP."""
    from .integration import harmony as _harmony
    from .neighbors import knn_and_fuzzy
    from .umap import umap as _umap

    t0 = time.perf_counter()
    if method == "bbknn":
        knn, conn = knn_and_fuzzy(
            adata.obsm["X_pca"], adata.obs["_batch"].to_numpy(),
            neighbors_within_batch=cfg.neighbors_within_batch,
            backend=cfg.bbknn_backend,
            metric=cfg.knn_metric,
        )
        embed_for_scib = adata.obsm["X_pca"]
        adata.obsm[f"X_pca_{method}"] = embed_for_scib
    elif method == "harmony":
        _harmony(
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
        h_info = adata.uns.get("harmony", {})
        converged = h_info.get("converged_at_iter", None)
        mode = (h_info.get("params", {}) or {}).get("lambda_mode", "?")
        if converged is not None:
            status = (f"converged at iter {converged}"
                      if converged < cfg.harmony_max_iter
                      else f"did NOT converge (max_iter={cfg.harmony_max_iter})")
            print(f"         [harmony] {status}  (lambda={mode}, "
                  f"theta={cfg.harmony_theta}, sigma={cfg.harmony_sigma})")
        embed_for_scib = adata.obsm["X_pca_harmony"]
        adata.obsm[f"X_pca_{method}"] = embed_for_scib
        dummy_batch = np.zeros(adata.n_obs, dtype=np.int32)
        knn, conn = knn_and_fuzzy(
            embed_for_scib, dummy_batch,
            neighbors_within_batch=cfg.knn_n_neighbors,
            backend=cfg.bbknn_backend,
            metric=cfg.knn_metric,
        )
    elif method == "none":
        dummy_batch = np.zeros(adata.n_obs, dtype=np.int32)
        knn, conn = knn_and_fuzzy(
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

    if cfg.run_umap:
        t0 = time.perf_counter()
        _run_umap_for_route(adata, method, conn, cfg)
        route_t["umap"] = _step(f"08 {method}/umap", t0) - t0

    return knn, conn, embed_for_scib


def _run_umap_for_route(adata, method: str, conn, cfg: PipelineConfig) -> None:
    """Route-owned UMAP layout written to ``obsm[X_umap_<method>]``."""
    from .umap import umap as _umap_fn

    key = f"{method}_connectivities"
    adata.obsp[key] = conn
    _umap_fn(
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


def _phase2_metrics_cluster(
    adata, method: str, cfg: PipelineConfig, route_t: dict[str, float],
    *, knn: dict, embed: np.ndarray, conn: Any,
) -> None:
    """Phase 2 — scIB + Leiden + cluster-homogeneity (ROGUE + SCCAF)."""
    from .cluster.resolution import auto_resolution

    if cfg.run_metrics:
        t0 = time.perf_counter()
        scib = _compute_scib_for_route(adata, method, knn, embed, cfg)
        adata.uns[f"scib_{method}"] = scib
        for k, v in scib.items():
            if isinstance(v, (int, float)):
                print(f"             {k:22s} = {v:.3f}")
        route_t["metrics"] = _step(f"09 {method}/metrics", t0) - t0

    if cfg.run_leiden:
        t0 = time.perf_counter()
        labels, chosen_res = auto_resolution(adata, method, conn, cfg)
        adata.obs[f"leiden_{method}"] = labels
        adata.uns[f"leiden_{method}_resolution"] = chosen_res
        print(f"         [{method}] picked r={chosen_res} → "
              f"{len(np.unique(labels))} clusters")
        route_t["leiden"] = _step(f"10 {method}/leiden", t0) - t0

    if cfg.run_metrics and cfg.run_leiden and cfg.compute_homogeneity:
        _compute_homogeneity_for_route(adata, method, embed, cfg, route_t)


def _compute_scib_for_route(
    adata, method: str, knn: dict, embedding: np.ndarray, cfg: PipelineConfig,
) -> dict[str, Any]:
    """scIB aggregation for this route's kNN + label pair."""
    from .scib_metrics import scib_score

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
    return scib_score(
        knn_idx, knn_dist,
        batch_labels=adata.obs["_batch"].to_numpy(),
        label_labels=label_arr,
        embedding=embed_for_scib,
    )


def _compute_homogeneity_for_route(
    adata, method: str, embed: np.ndarray, cfg: PipelineConfig,
    route_t: dict[str, float],
) -> None:
    """ROGUE + SCCAF for this route."""
    from .rogue import rogue_mean
    from .scib_metrics import sccaf_accuracy

    cluster_labels = adata.obs[f"leiden_{method}"].astype(str).to_numpy()
    sample_labels = adata.obs.get("_batch", None)
    if sample_labels is not None:
        sample_labels = sample_labels.astype(str).to_numpy()

    t0 = time.perf_counter()
    try:
        counts_gxc = (adata.layers["counts"]
                      if sp.issparse(adata.layers["counts"])
                      else sp.csr_matrix(adata.layers["counts"])).T
        rogue = rogue_mean(
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

    t0 = time.perf_counter()
    try:
        sccaf_acc = sccaf_accuracy(embed, cluster_labels)
        adata.uns[f"scib_{method}"]["sccaf"] = sccaf_acc
        print(f"             {'sccaf':22s} = {sccaf_acc:.3f}")
    except Exception as e:
        print(f"         [{method}] SCCAF failed: {type(e).__name__}: {e}")
    route_t["sccaf"] = _step(f"10b {method}/sccaf", t0) - t0

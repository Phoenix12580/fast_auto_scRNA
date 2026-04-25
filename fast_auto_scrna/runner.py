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

    # Pre-scIB cross-route UMAP comparison — so the user sees the big
    # side-by-side grid before the slow Phase 2 runs.
    if cfg.plot_dir and len(methods) > 1:
        from .plotting import compare_integration_plot
        plot_dir = Path(cfg.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = compare_integration_plot(
            adata, plot_dir / "integration_comparison.png",
            label_key=cfg.label_key if (cfg.label_key and cfg.label_key in adata.obs.columns) else "_batch",
        )
        print(f"\n[comparison-plot] (pre-scIB) → {plot_path}")

    # Phase 2a — scIB metrics for ALL routes (no Leiden yet). These are
    # label-free (iLISI, kBET) or use ct.main/label_key — cheap, does not
    # need the 150-point Leiden sweep. Lets us emit the cross-route scIB
    # heatmap BEFORE picking a winner, so the user can confirm the
    # auto-selection before paying the Leiden cost.
    _banner("Phase 2a: scIB metrics for all routes (no Leiden)")
    _phase2a_scib_all_routes(
        adata, methods, route_artifacts, cfg, route_timings,
    )

    # Emit the cross-route scIB heatmap BEFORE Phase 2b so the user sees
    # the integration comparison without waiting for Leiden.
    if len(methods) > 1 and cfg.plot_dir and cfg.run_metrics:
        from .plotting import compare_scib_heatmap
        plot_dir = Path(cfg.plot_dir)
        try:
            p = compare_scib_heatmap(
                adata, plot_dir / "scib_heatmap_pre_cluster.png",
                methods=methods,
            )
            print(f"\n[scib-heatmap-pre-cluster] → {p}")
        except Exception as e:
            print(f"[scib-heatmap-pre-cluster] failed: {type(e).__name__}: {e}")

    # Human-in-the-loop gate (multi-route + no explicit cluster_method):
    # Phase 2b spends ~10-30 min on Leiden + ROGUE + SCCAF for the winner.
    # The auto-pick (argmax scIB mean) can be wrong on atlases where the
    # mean weighting doesn't match dataset-specific priorities, so we stop
    # here, print the recommendation, and ask the user to re-run with
    # cfg.cluster_method=<route>. Single-route runs and runs that have
    # already pinned cfg.cluster_method skip the gate.
    if len(methods) > 1 and getattr(cfg, "cluster_method", None) is None:
        auto_pick, auto_score = _auto_pick_by_scib_mean(adata, methods)
        _banner("GATE: Phase 2a done — human decision required")
        print("\nPer-route scIB mean (Phase 2a):")
        for m in methods:
            sc = adata.uns.get(f"scib_{m}", {})
            score = sc.get("mean", float("nan"))
            if isinstance(score, (int, float)) and not np.isnan(score):
                marker = "  ← auto-pick" if m == auto_pick else ""
                print(f"  {m:10s} scib mean = {score:.4f}{marker}")
            else:
                print(f"  {m:10s} scib mean = n/a")
        if cfg.plot_dir:
            print(f"\nInspect: {Path(cfg.plot_dir) / 'scib_heatmap_pre_cluster.png'}")
        print("\nTo run Phase 2b (Leiden + ROGUE + SCCAF), re-run with:")
        print(f"  cfg.cluster_method = {auto_pick!r}   # accept auto-pick")
        print(f"  # or any of: {list(methods)}")

        adata.uns["fast_auto_scrna_gate_paused"] = True
        adata.uns["fast_auto_scrna_auto_pick"] = auto_pick
        for m, steps in route_timings.items():
            for k, v in steps.items():
                timings[f"{m}/{k}"] = v
        total = time.perf_counter() - pipeline_t0
        adata.uns["fast_auto_scrna_timings"] = timings
        _banner(f"Phase 1+2a done in {total:.1f}s ({total / 60:.1f} min) — gate paused")
        for k, v in timings.items():
            print(f"  {k:15s} {v:6.1f}s")
        print(f"  {'TOTAL':15s} {total:6.1f}s")
        return adata

    # Method selection: either user-specified (cfg.cluster_method) or
    # auto by best scIB mean from Phase 2a metrics.
    selected_method = _select_cluster_method(adata, methods, cfg)
    if len(methods) > 1:
        _banner(f"Phase 2b: Leiden + ROGUE + SCCAF for winner = {selected_method!r}")
    else:
        _banner(f"Phase 2b: Leiden + ROGUE + SCCAF for {selected_method!r}")

    arts = route_artifacts[selected_method]
    _phase2_metrics_cluster(
        adata, selected_method, cfg, route_timings[selected_method],
        knn=arts["knn"], embed=arts["embed"], conn=arts["conn"],
        run_cluster=True,
    )
    adata.uns["selected_method"] = selected_method

    # Phase 2c (v2-P10, opt-in): cluster non-winner routes at the winner's
    # chosen resolution, then ROGUE + SCCAF. Standard scIB benchmarking
    # approach — fixes the resolution to isolate the integration effect.
    if (cfg.cluster_non_winners_at_winner_res and len(methods) > 1
            and cfg.run_metrics and cfg.run_leiden):
        winner_res = adata.uns.get(f"leiden_{selected_method}_resolution")
        if winner_res is None:
            print(f"[phase2c] winner {selected_method!r} has no chosen "
                  f"resolution — skipping non-winner reclustering")
        else:
            _banner(
                f"Phase 2c: non-winner routes Leiden + ROGUE + SCCAF at "
                f"winner r={winner_res} (cluster_non_winners_at_winner_res=True)"
            )
            from .cluster import leiden as _leiden_call
            for method in methods:
                if method == selected_method:
                    continue
                arts_m = route_artifacts[method]
                conn_m, embed_m = arts_m["conn"], arts_m["embed"]
                route_t_m = route_timings[method]
                t0 = time.perf_counter()
                _leiden_call(
                    adata, resolution=float(winner_res),
                    key_added=f"leiden_{method}", adjacency=conn_m,
                )
                adata.uns[f"leiden_{method}_resolution"] = float(winner_res)
                adata.uns[f"leiden_{method}_resolution_source"] = (
                    f"copied from winner {selected_method!r}"
                )
                n_k = adata.obs[f"leiden_{method}"].nunique()
                print(f"  [{method}] r={winner_res} → {n_k} clusters")
                route_t_m["leiden"] = _step(f"10 {method}/leiden", t0) - t0
                if cfg.compute_homogeneity:
                    _compute_homogeneity_for_route(
                        adata, method, embed_m, cfg, route_t_m,
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

    # all-mode: assemble scIB comparison table + heatmap + rogue grid
    if len(methods) > 1:
        from .plotting import (
            scib_comparison_table, compare_scib_heatmap,
            compare_rogue_per_cluster,
        )
        _banner("scIB comparison across integration methods")
        table = scib_comparison_table(adata, methods)
        # Store as a DataFrame — anndata's h5ad serializer handles those
        # cleanly (list-of-dicts with mixed types + NaN kBET blows up on
        # vlen-string conversion).
        import pandas as pd
        adata.uns["scib_comparison"] = pd.DataFrame(table)
        for row in table:
            print("  " + "  ".join(f"{k}={row[k]}" for k in row))

        if cfg.plot_dir:
            plot_dir = Path(cfg.plot_dir)
            try:
                p = compare_scib_heatmap(
                    adata, plot_dir / "scib_heatmap.png", methods=methods,
                )
                print(f"\n[scib-heatmap] → {p}")
            except Exception as e:
                print(f"[scib-heatmap] failed: {type(e).__name__}: {e}")

            have_rogue = any(
                f"rogue_per_cluster_{m}" in adata.uns for m in methods
            )
            if have_rogue:
                try:
                    p = compare_rogue_per_cluster(
                        adata, plot_dir / "rogue_comparison.png", methods=methods,
                    )
                    print(f"[rogue-comparison] → {p}")
                except Exception as e:
                    print(f"[rogue-comparison] failed: {type(e).__name__}: {e}")

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
    from .integration import (
        harmony as _harmony, fastmnn as _fastmnn, scvi_train as _scvi_train,
    )
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
    elif method == "scvi":
        latent, scvi_info = _scvi_train(
            adata,
            batch_key="_batch",
            n_latent=cfg.scvi_n_latent,
            n_hidden=cfg.scvi_n_hidden,
            n_layers=cfg.scvi_n_layers,
            max_epochs=cfg.scvi_max_epochs,
            early_stopping=cfg.scvi_early_stopping,
            gene_likelihood=cfg.scvi_gene_likelihood,
            dispersion=cfg.scvi_dispersion,
            use_hvg=cfg.scvi_use_hvg,
            accelerator=cfg.scvi_accelerator,
            batch_size=cfg.scvi_batch_size,
            seed=cfg.scvi_seed,
        )
        adata.obsm["X_scvi"] = latent
        adata.obsm[f"X_pca_{method}"] = latent
        adata.uns["scvi"] = scvi_info
        embed_for_scib = latent
        print(f"         [scvi] latent={latent.shape}  "
              f"epochs={scvi_info['max_epochs']} accelerator={scvi_info['accelerator']} "
              f"trained_on {scvi_info['n_train_genes']} genes")
        dummy_batch = np.zeros(adata.n_obs, dtype=np.int32)
        knn, conn = knn_and_fuzzy(
            embed_for_scib, dummy_batch,
            neighbors_within_batch=cfg.knn_n_neighbors,
            backend=cfg.bbknn_backend,
            metric=cfg.knn_metric,
        )
    elif method == "fastmnn":
        batch_codes_int = np.asarray(adata.obs["_batch"].cat.codes
                                      if hasattr(adata.obs["_batch"], "cat")
                                      else adata.obs["_batch"])
        result = _fastmnn(
            adata.obsm["X_pca"].astype(np.float32, copy=False),
            batch_codes_int,
            n_neighbors=cfg.fastmnn_n_neighbors,
            sigma_scale=cfg.fastmnn_sigma_scale,
            n_threads=cfg.fastmnn_n_threads,
        )
        embed_for_scib = result["corrected"]
        adata.obsm[f"X_pca_{method}"] = embed_for_scib
        adata.uns["fastmnn"] = {
            "n_pairs_per_merge": result["n_pairs_per_merge"],
            "merge_order": result["merge_order"],
            "skipped_batches": [str(b) for b in result["skipped_batches"]],
        }
        if result["skipped_batches"]:
            print(f"         [fastmnn] WARNING: no MNN found for batches "
                  f"{result['skipped_batches']} (kept uncorrected)")
        print(f"         [fastmnn] merge_order={result['merge_order']} "
              f"n_pairs={result['n_pairs_per_merge']}")
        dummy_batch = np.zeros(adata.n_obs, dtype=np.int32)
        knn, conn = knn_and_fuzzy(
            embed_for_scib, dummy_batch,
            neighbors_within_batch=cfg.knn_n_neighbors,
            backend=cfg.bbknn_backend,
            metric=cfg.knn_metric,
        )
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


def _auto_pick_by_scib_mean(
    adata, methods: tuple[str, ...],
) -> tuple[str, float]:
    """Return ``(best_method, best_score)`` by argmax of ``scib_{m}["mean"]``.

    NaN / missing scores are skipped; if every method is missing a score, the
    first method is returned with score ``-inf`` so callers can detect the
    degenerate case.
    """
    best_method = methods[0]
    best_score = float("-inf")
    for m in methods:
        sc = adata.uns.get(f"scib_{m}", {})
        score = sc.get("mean", float("nan"))
        if isinstance(score, (int, float)) and not np.isnan(score):
            if score > best_score:
                best_score = score
                best_method = m
    return best_method, best_score


def _select_cluster_method(
    adata, methods: tuple[str, ...], cfg: PipelineConfig,
) -> str:
    """Pick the integration method to run Leiden on.

    Priority:
      1. ``cfg.cluster_method`` if set and valid.
      2. Best ``scib_{method}["mean"]`` across Phase 2a results.
      3. Fallback: first method.
    """
    requested = getattr(cfg, "cluster_method", None)
    if requested:
        if requested in methods:
            print(f"\n[select] user-specified cluster_method={requested!r}")
            return requested
        print(f"\n[select] cfg.cluster_method={requested!r} not in routes "
              f"{methods}; falling back to auto-select")

    if len(methods) == 1:
        return methods[0]

    best_method, best_score = _auto_pick_by_scib_mean(adata, methods)
    print(f"\n[select] auto-selecting by scIB mean across {len(methods)} routes:")
    for m in methods:
        sc = adata.uns.get(f"scib_{m}", {})
        score = sc.get("mean", float("nan"))
        if isinstance(score, (int, float)) and not np.isnan(score):
            print(f"  {m:10s} scib mean = {score:.4f}")
        else:
            print(f"  {m:10s} scib mean = n/a (skipped)")
    print(f"  → selected: {best_method!r} (scib mean = {best_score:.4f})")
    return best_method


def _phase2_metrics_cluster(
    adata, method: str, cfg: PipelineConfig, route_t: dict[str, float],
    *, knn: dict, embed: np.ndarray, conn: Any,
    run_cluster: bool = True,
) -> None:
    """Phase 2 — scIB + optional Leiden + cluster-homogeneity (ROGUE + SCCAF).

    ``run_cluster=False`` skips the Leiden sweep + ROGUE + SCCAF, leaving
    only the label-free scIB block. Used in "evaluate all routes, cluster
    winner only" mode (v2-P9) to avoid spending 20+ min of Leiden on
    routes that won't be selected.
    """
    from .cluster.resolution import auto_resolution

    if cfg.run_metrics:
        # v2-P12: skip recomputation when Phase 2a already produced an
        # equivalent scIB dict for this route. The two passes use the
        # same kNN + embed; the only label source that could differ is
        # ``leiden_{method}`` — which Phase 2a uses only when
        # ``cfg.label_key`` is unset (and Phase 2b would then have
        # genuinely new labels to score against). When ``label_key`` is
        # pinned (typical atlas runs), both passes use the same GT labels
        # → identical scib output, so we reuse the cached dict and save
        # ~9 min on 222k. Falls through to recompute when no Phase 2a
        # cache exists or label source is leiden.
        scib_key = f"scib_{method}"
        cached = adata.uns.get(scib_key)
        cache_reusable = (
            cached is not None
            and cfg.label_key
            and cfg.label_key in adata.obs.columns
        )
        if cache_reusable:
            scib = cached
            print(f"  [09 {method}/metrics] reusing Phase 2a cache "
                  f"(label_key={cfg.label_key!r})")
            route_t["metrics"] = 0.0
        else:
            t0 = time.perf_counter()
            scib = _compute_scib_for_route(adata, method, knn, cfg, embed=embed)
            adata.uns[scib_key] = scib
            route_t["metrics"] = _step(f"09 {method}/metrics", t0) - t0
        for k, v in scib.items():
            if isinstance(v, (int, float)):
                if np.isnan(v):
                    print(f"             {k:22s} = n/a")
                else:
                    print(f"             {k:22s} = {v:.3f}")
        if "kbet_note" in scib:
            print(f"             [kbet note] {scib['kbet_note']}")

    if run_cluster and cfg.run_leiden:
        t0 = time.perf_counter()
        labels, chosen_res = auto_resolution(adata, method, conn, cfg)
        adata.obs[f"leiden_{method}"] = labels
        adata.uns[f"leiden_{method}_resolution"] = chosen_res
        print(f"         [{method}] picked r={chosen_res} → "
              f"{len(np.unique(labels))} clusters")
        route_t["leiden"] = _step(f"10 {method}/leiden", t0) - t0

    if run_cluster and cfg.run_metrics and cfg.run_leiden and cfg.compute_homogeneity:
        _compute_homogeneity_for_route(adata, method, embed, cfg, route_t)


def _scib_worker_init(thread_count: int, priority: str | None) -> None:
    """Cap BLAS / OMP threads + lower OS priority before scib-metrics
    imports JAX. Runs once per worker on ProcessPoolExecutor startup.

    Limiting BLAS threads is critical: each worker would otherwise try
    to use all 24 cores, oversubscribing 4× and dropping throughput
    below the sequential baseline. Reading these env vars MUST happen
    before any numpy/JAX import in the worker — that's why this is the
    initializer (runs first on worker boot) rather than inside the
    worker function (runs after first task arrives).
    """
    import os
    val = str(max(1, thread_count))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = val
    if priority:
        # Lazy import — keeps cluster.resolution off the critical path
        # for single-route runs that never spin up the scib pool.
        from .cluster.resolution import _set_process_priority
        _set_process_priority(priority)


def _scib_worker(args: tuple) -> tuple:
    """Compute scIB for one route. Returns ``(method, scib_dict, wall_s)``.

    Module-level (not closure) so Windows spawn-based ProcessPoolExecutor
    can import it cleanly.
    """
    import time as _time
    method, inputs = args
    t0 = _time.perf_counter()
    scib = _run_scib_compute(inputs)
    return method, scib, _time.perf_counter() - t0


def _print_scib_block(method: str, scib: dict[str, Any]) -> None:
    """Per-route scIB summary block — same look as the sequential path."""
    for k, v in scib.items():
        if isinstance(v, (int, float)):
            if np.isnan(v):
                print(f"             {k:22s} = n/a")
            else:
                print(f"             {k:22s} = {v:.3f}")
    if "kbet_note" in scib:
        print(f"             [kbet note] {scib['kbet_note']}")


def _phase2a_scib_all_routes(
    adata, methods: tuple[str, ...], route_artifacts: dict, cfg: PipelineConfig,
    route_timings: dict[str, dict[str, float]],
) -> None:
    """Compute Phase 2a scIB metrics for all routes.

    Parallel via ProcessPoolExecutor when ``cfg.scib_parallel`` is on,
    silhouettes are enabled (the actual bottleneck), and there is more
    than one route. Otherwise falls back to the sequential per-route
    loop, which preserves single-route timing line-for-line.

    Per-metric results are bit-identical between the two paths — both
    call the same ``scib_metrics`` JAX kernels on the same input arrays.
    """
    if not cfg.run_metrics:
        return

    use_parallel = (
        cfg.scib_parallel
        and len(methods) > 1
        and cfg.compute_silhouette  # silhouettes are the only paid bottleneck
    )

    if not use_parallel:
        for method in methods:
            print(f"\n── route: {method} (phase 2a — scIB only) ──")
            arts = route_artifacts[method]
            t0 = time.perf_counter()
            scib = _compute_scib_for_route(
                adata, method, arts["knn"], cfg, embed=arts["embed"],
            )
            adata.uns[f"scib_{method}"] = scib
            _print_scib_block(method, scib)
            route_timings[method]["metrics"] = (
                _step(f"09 {method}/metrics", t0) - t0
            )
        return

    # Parallel path
    import os as _os
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as _mp

    cpu = _os.cpu_count() or 1
    if cfg.scib_max_workers is None:
        n_workers = max(1, min(len(methods), cpu - 4))
    else:
        n_workers = max(1, min(len(methods), int(cfg.scib_max_workers)))
    threads_per = max(1, cpu // n_workers)

    print(f"\n[phase2a-parallel] {len(methods)} routes × scIB → "
          f"{n_workers} workers × {threads_per} BLAS threads each")

    tasks = []
    for method in methods:
        arts = route_artifacts[method]
        inputs = _prepare_scib_inputs(
            adata, method, arts["knn"], cfg, embed=arts["embed"],
        )
        tasks.append((method, inputs))

    ctx = _mp.get_context("spawn")
    wall_t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_scib_worker_init,
        initargs=(threads_per, cfg.leiden_worker_priority),
    ) as ex:
        # ex.map preserves submission order, so per-route output appears
        # in the same sequence as the sequential path. The first result
        # blocks until that route finishes, which on a balanced 4-route
        # run is also roughly when the rest finish.
        for method, scib, wall in ex.map(_scib_worker, tasks):
            print(f"\n── route: {method} (phase 2a — scIB only) ──")
            adata.uns[f"scib_{method}"] = scib
            _print_scib_block(method, scib)
            print(f"  [09 {method}/metrics] {wall:.1f}s (worker)")
            route_timings[method]["metrics"] = wall

    wall_total = time.perf_counter() - wall_t0
    seq_sum = sum(route_timings[m].get("metrics", 0.0) for m in methods)
    if seq_sum > 0:
        speedup = seq_sum / wall_total
        print(f"\n[phase2a-parallel] wall {wall_total:.1f}s vs Σ worker "
              f"{seq_sum:.1f}s → {speedup:.2f}× speedup")


def _prepare_scib_inputs(
    adata, method: str, knn: dict, cfg: PipelineConfig,
    *, embed: np.ndarray | None = None,
) -> dict[str, Any]:
    """Touch ``adata`` once and emit a pure-array bundle that
    ``_run_scib_compute`` (or its worker counterpart) can consume.

    Split out from ``_compute_scib_for_route`` at v2-P11 so the same input
    bundle can ship to a ``ProcessPoolExecutor`` worker without pickling
    the full AnnData.
    """
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

    return {
        "knn_idx": knn_idx,
        "knn_dist": knn_dist,
        "batch_labels": adata.obs["_batch"].to_numpy(),
        "label_labels": label_arr,
        "embedding": embed if cfg.compute_silhouette else None,
        "compute_kbet": cfg.compute_kbet,
    }


def _run_scib_compute(inputs: dict[str, Any]) -> dict[str, Any]:
    """Pure-array scIB compute. Same code path in main process (sequential)
    and inside ProcessPoolExecutor workers (parallel)."""
    from .scib_metrics import scib_score

    return scib_score(
        inputs["knn_idx"], inputs["knn_dist"],
        batch_labels=inputs["batch_labels"],
        label_labels=inputs["label_labels"],
        embedding=inputs["embedding"],
        compute_kbet=inputs["compute_kbet"],
    )


def _compute_scib_for_route(
    adata, method: str, knn: dict, cfg: PipelineConfig,
    *, embed: np.ndarray | None = None,
) -> dict[str, Any]:
    """scIB aggregation for this route's kNN + label pair.

    When ``cfg.compute_silhouette`` is True and ``embed`` is provided, the
    3 embedding silhouettes (label / batch / isolated) are added via
    scib-metrics' JAX backend — adds ~5 min per route on 222k cells.
    """
    inputs = _prepare_scib_inputs(adata, method, knn, cfg, embed=embed)
    return _run_scib_compute(inputs)


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

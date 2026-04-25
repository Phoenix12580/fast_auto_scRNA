"""Atlas-scale smoke: 222 529 cells × 20 055 genes, 10 batches.

Ported from the v1 222k smoke config. Matches the ROADMAP performance
baseline so v2 can be compared head-to-head.

Notes
-----
- ``hvg_flavor='seurat'`` (not ``'seurat_v3'``) — v1 discovered that
  scikit-misc loess segfaults silently at 222k × 10 batches on seurat_v3.
- ``batch_key='data.sets'`` — 10 logical batches (``orig.ident`` is
  per-sample, 75 unique, too granular).
- ``label_key='ct.main'`` — 3-class ground truth, the coarsest lineage
  layer (epithelia / immune / stromal).
- ``silhouette_n_iter=50`` — same as v1 baseline (default 100 would
  double the 890 s sklearn sweep).

Expected wall ~18 min until GS-3 (Rust silhouette kernel) lands, then
~4 min.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/StepF.All_Cells.h5ad",
        help="h5ad path (default: data/StepF.All_Cells.h5ad)",
    )
    parser.add_argument(
        "--out", default="benchmarks/out/smoke_222k_v2p10.h5ad",
        help="where to write the processed AnnData",
    )
    parser.add_argument(
        "--plot-dir", default="benchmarks/out/smoke_222k_v2p10_plots",
        help="directory for per-route plots (set '' to skip)",
    )
    parser.add_argument(
        "--integration", default="all",
        choices=["bbknn", "harmony", "fastmnn", "scvi", "all"],
        help="'all' runs bbknn / harmony / fastmnn / scvi and produces "
             "cross-route comparison plots (default) — use this for the "
             "first run on any new atlas to pick the best method. The "
             "'none' baseline route was removed 2026-04-25.",
    )
    parser.add_argument(
        "--cluster-method", default=None,
        choices=[None, "bbknn", "harmony", "fastmnn", "scvi"],
        help="When --integration=all, set this to skip the human-decision "
             "gate and immediately run Phase 2b (Leiden + ROGUE + SCCAF) "
             "for the chosen route. Leave None to early-exit at the gate.",
    )
    parser.add_argument(
        "--cluster-non-winners-at-winner-res", action="store_true",
        help="After winner Phase 2b, cluster non-winner routes at the "
             "winner's chosen resolution + ROGUE + SCCAF. Adds ~12 min "
             "on 222k for 3 non-winners (vs ~90 min for independent sweeps).",
    )
    parser.add_argument("--silhouette-n-iter", type=int, default=50)
    args = parser.parse_args()

    from fast_auto_scrna import run_pipeline

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    adata = run_pipeline(
        input_h5ad=args.input,
        batch_key="data.sets",
        label_key="ct.main",
        integration=args.integration,
        cluster_method=args.cluster_method,
        cluster_non_winners_at_winner_res=args.cluster_non_winners_at_winner_res,
        hvg_flavor="seurat",             # seurat_v3 segfaults at 222k × 10-batch
        hvg_n_top_genes=2000,
        pca_n_comps="auto",              # Gavish-Donoho
        silhouette_n_iter=args.silhouette_n_iter,
        compute_homogeneity=True,
        plot_dir=args.plot_dir or None,
        out_h5ad=str(out_path),
    )
    wall = time.perf_counter() - t0

    print()
    print("=" * 72)
    print(f"222k smoke done — total wall {wall:.1f} s ({wall / 60:.1f} min)")
    print("=" * 72)

    if adata.uns.get("fast_auto_scrna_gate_paused"):
        ap = adata.uns.get("fast_auto_scrna_auto_pick", "?")
        print(f"\n[gate] Phase 2a done — re-run with --cluster-method {ap} "
              "to run Phase 2b for the auto-picked winner.")
        method = ap
    else:
        method = args.integration if args.integration != "all" else (
            args.cluster_method or "bbknn"
        )
    scib = adata.uns.get(f"scib_{method}", {})
    for k in ("ilisi", "clisi", "graph_connectivity", "kbet_acceptance",
              "label_silhouette", "batch_silhouette", "isolated_label",
              "rogue_mean", "sccaf", "mean"):
        if k in scib:
            print(f"  {k:22s} = {scib[k]:.4f}")

    leiden_key = f"leiden_{method}"
    if leiden_key in adata.obs.columns:
        n_k = adata.obs[leiden_key].nunique()
        print(f"  {'leiden clusters':22s} = {n_k}")

    # Resolution-selection curve: knee (v2-P8) > conductance (v2-P7) > silhouette (legacy).
    knee_key = f"knee_curve_{method}"
    cond_key = f"conductance_curve_{method}"
    silh_key = f"silhouette_curve_{method}"
    if knee_key in adata.uns:
        c = adata.uns[knee_key]
        knee_r   = next(r for r, f in zip(c["resolution"], c["is_knee"])   if f)
        picked_r = next(r for r, f in zip(c["resolution"], c["is_picked"]) if f)
        print(f"  knee curve (knee r={knee_r:.2f}, picked r={picked_r:.2f}):")
        for r, k, cd, ik, ip in zip(c["resolution"], c["n_clusters"],
                                     c["conductance"], c["is_knee"], c["is_picked"]):
            tag = ""
            if ik: tag += "knee "
            if ip: tag += "★PICKED"
            if tag:
                print(f"    r={r:.2f}  k={k:2d}  cond={cd:.4f}  {tag}")
    elif cond_key in adata.uns:
        curve = adata.uns[cond_key]
        print(f"  conductance curve (res → k → cond, min=tightest):")
        for r, k, c in zip(
            curve["resolution"], curve["n_clusters"], curve["conductance"]
        ):
            print(f"    r={r:.2f}  k={k:2d}  cond={c:.4f}")
    elif silh_key in adata.uns:
        curve = adata.uns[silh_key]
        print(f"  silhouette curve (res → k → silhouette):")
        for r, k, s in zip(
            curve["resolution"], curve["n_clusters"], curve["mean_silhouette"]
        ):
            print(f"    r={r:.2f}  k={k:2d}  s={s:+.5f}")


if __name__ == "__main__":
    main()

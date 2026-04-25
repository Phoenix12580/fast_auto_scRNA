"""Re-cluster non-winner routes on the v2p10 baseline h5ad.

Strategy: use the WINNER's selected resolution (from
``adata.uns[f"leiden_{winner}_resolution"]``) as the fixed resolution
for non-winner routes — single Leiden call per route, no 150-point
sweep. Compares all 4 methods at the same resolution, ~12 min total
(vs ~90 min if we ran independent sweeps per route).

Rationale: when comparing integration methods, fixing the resolution
isolates the effect of the integration itself. Different methods
producing different "best" k makes per-method-optimized comparison
confounded by k-choice. Winner-resolution comparison is the standard
benchmarking approach.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="benchmarks/out/smoke_222k_v2p10.h5ad")
    ap.add_argument("--out", default=None,
                    help="output h5ad (default: overwrite input)")
    ap.add_argument("--winner", default="bbknn",
                    help="winner route whose chosen resolution to copy")
    ap.add_argument("--routes", nargs="+",
                    default=["harmony", "fastmnn", "scvi"],
                    help="non-winner routes to recluster")
    ap.add_argument("--plot-dir", default="benchmarks/out/smoke_222k_v2p10_plots")
    args = ap.parse_args()

    out_path = args.out or args.h5ad

    print(f"loading {args.h5ad} ...")
    import anndata as ad
    t0 = time.perf_counter()
    adata = ad.read_h5ad(args.h5ad)
    print(f"  loaded {adata.n_obs} × {adata.n_vars} in "
          f"{time.perf_counter() - t0:.1f}s")

    from fast_auto_scrna.config import PipelineConfig
    from fast_auto_scrna.cluster import leiden as _leiden_call
    from fast_auto_scrna.runner import _compute_homogeneity_for_route

    cfg = PipelineConfig(
        input_h5ad=args.h5ad,
        batch_key="data.sets",
        label_key="ct.main",
        hvg_flavor="seurat",
        hvg_n_top_genes=2000,
        compute_homogeneity=True,
        run_metrics=True,
        run_leiden=True,
    )

    winner_res_key = f"leiden_{args.winner}_resolution"
    if winner_res_key not in adata.uns:
        print(f"[ERROR] winner resolution not found at adata.uns[{winner_res_key!r}]; "
              f"available: {[k for k in adata.uns if k.startswith('leiden_')]}")
        return 1
    winner_res = float(adata.uns[winner_res_key])
    winner_n_k = adata.obs[f"leiden_{args.winner}"].nunique() if f"leiden_{args.winner}" in adata.obs else "?"
    print(f"\nWinner = {args.winner!r}  resolution = {winner_res}  k = {winner_n_k}")
    print(f"Reclustering non-winners {args.routes} at this fixed resolution.")

    pipeline_t0 = time.perf_counter()
    for method in args.routes:
        print(f"\n{'=' * 72}\nRoute: {method}\n{'=' * 72}")

        if f"leiden_{method}" in adata.obs.columns:
            print(f"  [{method}] leiden_{method} already present — skipping")
            continue
        conn_key = f"{method}_connectivities"
        embed_key = f"X_pca_{method}"
        if conn_key not in adata.obsp:
            print(f"  [{method}] missing {conn_key} in obsp — skipping")
            continue
        if embed_key not in adata.obsm:
            print(f"  [{method}] missing {embed_key} in obsm — skipping")
            continue

        conn = adata.obsp[conn_key]
        embed = adata.obsm[embed_key]
        route_t: dict[str, float] = {}

        # Single Leiden call at winner's resolution.
        t0 = time.perf_counter()
        _leiden_call(
            adata, resolution=winner_res, key_added=f"leiden_{method}",
            adjacency=conn,
        )
        adata.uns[f"leiden_{method}_resolution"] = winner_res
        adata.uns[f"leiden_{method}_resolution_source"] = (
            f"copied from winner {args.winner!r}"
        )
        labels = adata.obs[f"leiden_{method}"].astype(str).to_numpy()
        n_k = len(np.unique(labels))
        print(f"  [{method}] r={winner_res} (from winner) → {n_k} clusters  "
              f"[{time.perf_counter() - t0:.1f}s]")

        # ROGUE + SCCAF (writes into adata.uns[scib_{method}])
        _compute_homogeneity_for_route(adata, method, embed, cfg, route_t)

    total_pipeline = time.perf_counter() - pipeline_t0
    print(f"\n{'=' * 72}")
    print(f"recluster done in {total_pipeline:.1f}s ({total_pipeline / 60:.1f} min)")
    print(f"{'=' * 72}")

    # Save back
    print(f"\nsaving {out_path} ...")
    t0 = time.perf_counter()
    adata.write_h5ad(out_path, compression="lzf")
    print(f"  wrote in {time.perf_counter() - t0:.1f}s")

    # Re-emit cross-route plots so scib_heatmap.png includes all 4 routes' rogue/sccaf
    if args.plot_dir:
        from fast_auto_scrna.plotting import (
            scib_comparison_table, compare_scib_heatmap,
            compare_rogue_per_cluster,
        )
        import pandas as pd

        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        methods = ("bbknn", "harmony", "fastmnn", "scvi")

        print(f"\nre-emitting comparison plots → {plot_dir}")
        table = scib_comparison_table(adata, methods)
        adata.uns["scib_comparison"] = pd.DataFrame(table)
        for row in table:
            print("  " + "  ".join(f"{k}={row[k]}" for k in row))

        try:
            p = compare_scib_heatmap(
                adata, plot_dir / "scib_heatmap.png", methods=methods,
            )
            print(f"  [scib-heatmap] → {p}")
        except Exception as e:
            print(f"  [scib-heatmap] failed: {type(e).__name__}: {e}")

        try:
            p = compare_rogue_per_cluster(
                adata, plot_dir / "rogue_comparison.png", methods=methods,
            )
            print(f"  [rogue-comparison] → {p}")
        except Exception as e:
            print(f"  [rogue-comparison] failed: {type(e).__name__}: {e}")

        # Resave with updated scib_comparison
        adata.write_h5ad(out_path, compression="lzf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

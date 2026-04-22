"""Smoke test: graph-silhouette resolution optimizer on 222k StepF atlas.

Loads StepF.All_Cells.h5ad (222529 x 20055, 10 batches, ct.main 3-class /
ct.sub 7-class / ct.sub.epi 13-class), runs the pipeline through BBKNN
(no UMAP, no Leiden — we'll do Leiden inside the optimizer), then scans
resolutions and plots the silhouette curve.

Goal: confirm the 222k BBKNN graph produces a clean silhouette peak in
the atlas-scale k∈[3,10] band (expected to land near ct.sub's 7 classes).

Run:
  wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1 && \\
      /mnt/f/NMF_rewrite/scatlas/.venv/bin/python \\
      scatlas_pipeline/benchmarks/silhouette_smoke_222k.py"
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


H5 = Path("/mnt/f/NMF_rewrite/StepF.All_Cells.h5ad")
OUT = Path(
    "/mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scatlas_pipeline/benchmarks/"
    "silhouette_smoke_222k_out"
)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from scatlas_pipeline.pipeline import PipelineConfig, run_from_config
    from scatlas_pipeline.silhouette import (
        optimize_resolution_graph_silhouette,
        pick_best_resolution,
        plot_silhouette_curve,
    )

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] loading {H5} ...")
    adata = anndata.read_h5ad(H5)
    print(f"[smoke] loaded: {adata.shape}; obs cols: {list(adata.obs.columns)}")
    for col in ("ct.main", "ct.sub"):
        if col in adata.obs.columns:
            vc = adata.obs[col].value_counts()
            print(f"[smoke]   {col}: {dict(vc)}")

    # Pipeline through BBKNN — skip UMAP/Leiden (we do Leiden in the optimizer).
    cfg = PipelineConfig(
        input_h5ad=str(H5),
        batch_key="data.sets",
        integration="bbknn",
        hvg_flavor="seurat",          # seurat_v3 loess segfaulted on this dataset
        hvg_batch_aware=False,
        run_umap=False,
        run_leiden=False,             # optimizer runs Leiden internally
        compute_silhouette=False,
        compute_homogeneity=False,
        run_recall=False,
    )

    pre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    adata = run_from_config(cfg, adata_in=adata)
    t_pipe = time.perf_counter() - t0
    post = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\n[smoke] pipeline→BBKNN wall: {t_pipe:.1f}s "
          f"(+{(post - pre) / 1024 / 1024:.1f} GB RSS)")

    # Verify graph is there
    if "bbknn_connectivities" not in adata.obsp:
        raise RuntimeError(
            f"missing bbknn_connectivities; obsp keys: {list(adata.obsp.keys())}"
        )
    G = adata.obsp["bbknn_connectivities"]
    print(f"[smoke] bbknn graph: {G.shape}, nnz={G.nnz}, "
          f"avg degree={G.nnz / G.shape[0]:.1f}")

    # Atlas-scale resolution grid: wide but shallow on the low end (3-10 k target)
    resolutions = [round(r, 3) for r in np.arange(0.05, 0.85, 0.05)]

    t0 = time.perf_counter()
    curve = optimize_resolution_graph_silhouette(
        adata,
        method="bbknn",
        resolutions=resolutions,
        n_subsample=1000,
        n_iter=50,                    # 50 iters: tight enough SD, reasonable wall
        stratify_key="data.sets",     # balance across 10 batches
        seed=0,
        verbose=True,
    )
    t_opt = time.perf_counter() - t0
    print(f"\n[smoke] optimizer wall: {t_opt:.1f}s")

    print("\n[smoke] silhouette curve:")
    print(curve.to_string(index=False))

    # Pick best in atlas-scale band [3, 10]
    try:
        best_res = pick_best_resolution(curve, k_lo=3, k_hi=10)
        best_row = curve.loc[(curve["resolution"] - best_res).abs().idxmin()]
        print(
            f"\n[smoke] BEST in k∈[3,10]: res={best_res:.3f}, "
            f"k={int(best_row['n_clusters'])}, "
            f"silhouette={best_row['mean_silhouette']:.4f}"
            f" ± {best_row['sd_silhouette']:.4f}"
        )
    except ValueError as e:
        print(f"\n[smoke] WARN: {e}")
        best_res = None

    # ---- Plot + save artifacts --------------------------------------------
    curve.to_csv(OUT / "curve.csv", index=False)
    plot_silhouette_curve(
        curve,
        OUT / "curve.png",
        best_resolution=best_res,
        title=f"Graph silhouette on 222k StepF (BBKNN, stratified by data.sets)",
        k_lo=3, k_hi=10,
    )
    print(f"[smoke] curve CSV → {OUT / 'curve.csv'}")
    print(f"[smoke] plot PNG  → {OUT / 'curve.png'}")

    # Also dump a summary json
    summary = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "graph_nnz": int(G.nnz),
        "graph_avg_degree": float(G.nnz / G.shape[0]),
        "pipeline_wall_s": t_pipe,
        "optimizer_wall_s": t_opt,
        "n_iter": 50,
        "n_subsample": 1000,
        "resolutions": resolutions,
        "curve": curve.to_dict(orient="records"),
        "best_resolution": best_res,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[smoke] summary → {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()

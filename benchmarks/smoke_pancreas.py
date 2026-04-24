"""Small-data smoke: pancreas_sub (1000 cells × 15998 genes, 8 SubCellTypes).

Single-batch dataset (orig.ident has one value), so integration methods
are pass-through. Used to sanity-check the conductance resolution picker
on a dataset with FINER biology than 222k prostate (which had only 3
ct.main classes). Here SubCellType has 8 classes — genuine test of
whether the picker can hit the right k.

If conductance picks k ≈ 8 matching SubCellType → picker works across
dataset granularities. If it still picks r=0.05 / k=2-3 → the "monotone
favours smallest r" failure mode is general.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import time
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["CellType", "SubCellType"],
                        default="SubCellType")
    parser.add_argument("--out", default="benchmarks/out/smoke_pancreas.h5ad")
    parser.add_argument("--plot-dir", default="benchmarks/out/smoke_pancreas_plots")
    args = parser.parse_args()

    from fast_auto_scrna import run_pipeline

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    adata = run_pipeline(
        input_h5ad="data/pancreas_sub.h5ad",
        batch_key="orig.ident",      # single-value, integration is a no-op
        label_key=args.label,
        integration="none",          # single batch, nothing to integrate
        hvg_flavor="seurat",
        hvg_n_top_genes=2000,
        pca_n_comps="auto",
        plot_dir=args.plot_dir or None,
        out_h5ad=str(out_path),
        # use default 0.01-step grid + knee picker (cfg default)
    )
    wall = time.perf_counter() - t0

    print()
    print("=" * 72)
    print(f"pancreas smoke done — total wall {wall:.1f}s")
    print("=" * 72)

    scib = adata.uns.get("scib_none", {})
    for k in ("ilisi", "clisi", "graph_connectivity", "kbet_acceptance",
             "rogue_mean", "sccaf", "mean"):
        if k in scib:
            v = scib[k]
            print(f"  {k:22s} = {v if isinstance(v, str) else f'{v:.4f}'}")

    leiden_key = "leiden_none"
    if leiden_key in adata.obs.columns:
        print(f"  {'leiden clusters':22s} = {adata.obs[leiden_key].nunique()}")

    import pandas as pd
    knee_key = "knee_curve_none"
    cond_key = "conductance_curve_none"
    if knee_key in adata.uns:
        c = adata.uns[knee_key]
        knee_r   = next(r for r, f in zip(c["resolution"], c["is_knee"])   if f)
        picked_r = next(r for r, f in zip(c["resolution"], c["is_picked"]) if f)
        print(f"\nknee curve (knee r={knee_r:.2f}, picked r={picked_r:.2f}):")
        for r, k, cd, ik, ip in zip(c["resolution"], c["n_clusters"],
                                     c["conductance"], c["is_knee"], c["is_picked"]):
            tag = ""
            if ik: tag += "knee "
            if ip: tag += "★PICKED"
            if tag:
                print(f"  r={r:.2f}  k={k:2d}  cond={cd:.4f}   {tag}")
    elif cond_key in adata.uns:
        print(f"\nconductance curve (lower=tighter):")
        curve = adata.uns[cond_key]
        for r, k, c in zip(curve["resolution"], curve["n_clusters"],
                           curve["conductance"]):
            print(f"  r={r:.2f}  k={k:2d}  cond={c:.4f}")

    # ARI vs ground-truth — only a few anchor resolutions (grid is 150 points)
    from sklearn.metrics import adjusted_rand_score
    gt = adata.obs[args.label].astype(str).to_numpy()
    print(f"\nARI vs {args.label} (n_classes={len(set(gt))}):")
    anchors = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50]
    for r in anchors:
        col = f"leiden_none_r{r:.2f}"
        if col not in adata.obs.columns:
            continue
        lbl = adata.obs[col].astype(int).to_numpy()
        k = len(set(lbl.tolist()))
        ari = adjusted_rand_score(gt, lbl)
        print(f"  r={r:.2f}  k={k:3d}  ARI={ari:.4f}")


if __name__ == "__main__":
    main()

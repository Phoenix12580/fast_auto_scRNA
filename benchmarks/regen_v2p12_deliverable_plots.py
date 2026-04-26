"""Regenerate the v2-P12 atlas-222k deliverable plots in dual format
(PDF + PNG, scatter rasterized for Illustrator-friendly file sizes).

Inputs (cached):
  benchmarks/out/smoke_222k_v2p10.h5ad

For each route in INTEGRATION_METHODS that doesn't already have a CHAMP
curve cached in adata.uns, runs CHAMP once and persists labels + curve
back to the h5ad. So a second invocation skips the CHAMP step entirely.

Outputs (PDF + PNG dual format):
  benchmarks/out/v2p12_plots/
    01_integration_umap_comparison.{pdf,png}     4-route UMAP grid
    02_scib_heatmap.{pdf,png}                    methods × metrics
    03_rogue_per_cluster.{pdf,png}               cross-route ROGUE
    04_champ_landscape_grid.{pdf,png}            4-route CHAMP landscape
    05_picker_umap_per_route.{pdf,png}           knee vs CHAMP UMAP × 4 routes
    champ_landscape_<route>.{pdf,png}            per-route CHAMP (4 × 2)
"""
from __future__ import annotations

import time
from pathlib import Path

H5AD = "benchmarks/out/smoke_222k_v2p10.h5ad"
OUT = Path("benchmarks/out/v2p12_plots")


def _ensure_champ_for_all_routes(adata):
    """Run CHAMP on routes lacking cached curve; persist back to h5ad."""
    from fast_auto_scrna.cluster.resolution import auto_resolution
    from fast_auto_scrna.config import PipelineConfig, INTEGRATION_METHODS

    cfg = PipelineConfig(
        input_h5ad="<cached>",
        leiden_n_iterations=2,
        max_leiden_workers=None,
        leiden_worker_priority="below_normal",
    )

    dirty = False
    for m in INTEGRATION_METHODS:
        conn_key = f"{m}_connectivities"
        champ_obs_key = f"leiden_{m}_champ"
        champ_uns_key = f"champ_curve_{m}"
        if conn_key not in adata.obsp:
            print(f"[skip] {m}: no obsp[{conn_key}]")
            continue
        if champ_uns_key in adata.uns and champ_obs_key in adata.obs.columns:
            cur_k = adata.obs[champ_obs_key].nunique()
            print(f"[cache hit] {m}: champ already computed (k={cur_k})")
            continue
        print(f"[compute] CHAMP on {m}...")
        t0 = time.perf_counter()
        labels, picked_r = auto_resolution(adata, m, adata.obsp[conn_key], cfg)
        adata.obs[champ_obs_key] = labels.astype(str)
        adata.uns[f"{champ_obs_key}_resolution"] = picked_r
        dt = time.perf_counter() - t0
        print(f"  {m}: γ={picked_r:.3f} k={len(set(labels.tolist()))} ({dt:.0f}s)")
        dirty = True

    if dirty:
        print(f"[persist] writing CHAMP curves + labels back to {H5AD}")
        t0 = time.perf_counter()
        adata.write_h5ad(H5AD, compression="lzf")
        print(f"  wrote ({time.perf_counter()-t0:.0f}s)")


def main() -> int:
    import anndata as ad

    print(f"[load] {H5AD}")
    t0 = time.perf_counter()
    adata = ad.read_h5ad(H5AD)
    print(f"  shape {adata.shape}  ({time.perf_counter()-t0:.0f}s)")

    _ensure_champ_for_all_routes(adata)

    OUT.mkdir(parents=True, exist_ok=True)

    from fast_auto_scrna.plotting import (
        compare_integration_plot,
        compare_scib_heatmap,
        compare_rogue_per_cluster,
        compare_champ_landscape,
        compare_picker_umap,
        plot_champ_curve,
    )
    import pandas as pd

    print("\n=== plotting (PDF + PNG dual format) ===")

    print("[01] integration UMAP comparison...")
    p = compare_integration_plot(
        adata, OUT / "01_integration_umap_comparison.pdf",
        label_key=("ct.main" if "ct.main" in adata.obs.columns else None),
    )
    print(f"  → {p}")

    print("[02] scIB heatmap...")
    p = compare_scib_heatmap(adata, OUT / "02_scib_heatmap.pdf")
    print(f"  → {p}")

    if any(f"rogue_per_cluster_{m}" in adata.uns for m in
           ("bbknn", "harmony", "fastmnn", "scvi")):
        print("[03] ROGUE per-cluster comparison...")
        p = compare_rogue_per_cluster(adata, OUT / "03_rogue_per_cluster.pdf")
        print(f"  → {p}")
    else:
        print("[03] ROGUE: no rogue_per_cluster_<m> entries — skipping")

    print("[04] CHAMP landscape grid (4 routes)...")
    p = compare_champ_landscape(adata, OUT / "04_champ_landscape_grid.pdf")
    print(f"  → {p}")

    print("[05] picker UMAP per route (knee vs CHAMP)...")
    p = compare_picker_umap(adata, OUT / "05_picker_umap_per_route.pdf")
    print(f"  → {p}")

    print("[06] per-route CHAMP landscapes...")
    for m in ("bbknn", "harmony", "fastmnn", "scvi"):
        if f"champ_curve_{m}" in adata.uns:
            curve = pd.DataFrame(adata.uns[f"champ_curve_{m}"])
            p = plot_champ_curve(
                curve, OUT / f"champ_landscape_{m}.pdf",
                title=f"{m} — CHAMP modularity landscape",
            )
            print(f"  → {p}")

    print(f"\n[done] all plots in {OUT}")
    print("file inventory:")
    for f in sorted(OUT.glob("*")):
        kb = f.stat().st_size / 1024
        print(f"  {f.name:50s}  {kb:>8.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

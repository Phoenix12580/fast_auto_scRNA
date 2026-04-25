"""Side-by-side UMAP: GT cell types vs baseline knee leiden vs CHAMP leiden.

Cache-aware: writes ``leiden_bbknn_champ`` (and ``leiden_bbknn_champ_resolution``
in uns) BACK to the h5ad on first run. Subsequent invocations skip the
leiden step entirely and just plot — the h5ad is the cache.

Output: benchmarks/out/champ_vs_knee_222k.png
"""
from __future__ import annotations

import time
from pathlib import Path


CHAMP_GAMMA = 0.150  # picked by bench_phase2b_champ_222k (log-width=1.104)
CHAMP_KEY = "leiden_bbknn_champ"


def _ensure_champ_artifacts(h5ad_path: str):
    """Open h5ad. If the CHAMP partition + landscape curve aren't already
    cached, run CHAMP once (~5 min on 222k) and persist BOTH the chosen
    labels and the full hull-curve dict back to disk. Subsequent calls
    reuse the cache.
    """
    import anndata as ad
    from fast_auto_scrna.cluster.resolution import auto_resolution
    from fast_auto_scrna.config import PipelineConfig

    print(f"[load] {h5ad_path}")
    t0 = time.perf_counter()
    adata = ad.read_h5ad(h5ad_path)
    print(f"  shape {adata.shape}  ({time.perf_counter()-t0:.1f}s)")

    have_labels = CHAMP_KEY in adata.obs.columns
    have_curve = "champ_curve_bbknn" in adata.uns
    if have_labels and have_curve:
        print(f"[cache hit] {CHAMP_KEY} + champ_curve_bbknn present "
              f"(k={adata.obs[CHAMP_KEY].nunique()}) — skipping CHAMP run")
        return adata

    print(f"[cache miss] running CHAMP on bbknn graph "
          f"({'labels' if not have_labels else ''}"
          f"{' + ' if not have_labels and not have_curve else ''}"
          f"{'curve' if not have_curve else ''} missing)")
    cfg = PipelineConfig(
        input_h5ad="<cached>",
        leiden_n_iterations=2,
        max_leiden_workers=None,
        leiden_worker_priority="below_normal",
    )
    conn = adata.obsp["bbknn_connectivities"]
    t0 = time.perf_counter()
    labels, picked_r = auto_resolution(adata, "bbknn", conn, cfg)
    print(f"  CHAMP picked γ={picked_r:.3f} (k={len(set(labels.tolist()))}) "
          f"({time.perf_counter()-t0:.1f}s)")
    adata.obs[CHAMP_KEY] = labels.astype(str)
    adata.uns[f"{CHAMP_KEY}_resolution"] = picked_r

    print(f"[persist] writing CHAMP labels + curve back to {h5ad_path}")
    t0 = time.perf_counter()
    adata.write_h5ad(h5ad_path, compression="lzf")
    print(f"  wrote ({time.perf_counter()-t0:.1f}s) — next run skips CHAMP")
    return adata


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from fast_auto_scrna.cluster.resolution import plot_champ_curve

    adata = _ensure_champ_artifacts("benchmarks/out/smoke_222k_v2p10.h5ad")
    champ_labels = adata.obs[CHAMP_KEY].astype(str).values
    champ_r = float(adata.uns.get(f"{CHAMP_KEY}_resolution", CHAMP_GAMMA))

    # Auto-output the CHAMP modularity landscape (matches what the
    # pipeline emits via emit_route_plots when plot_dir is set).
    if "champ_curve_bbknn" in adata.uns:
        print("\n[plot] CHAMP modularity landscape...")
        curve = pd.DataFrame(adata.uns["champ_curve_bbknn"])
        landscape_out = Path("benchmarks/out/champ_landscape_222k.png")
        plot_champ_curve(
            curve, landscape_out,
            title="222k bbknn — CHAMP modularity landscape (Weir 2017)",
        )
        print(f"  → {landscape_out}")
    # Plot 2x2 grid
    print("\n[plot] building 2x2 UMAP comparison...")
    X = adata.obsm["X_umap_bbknn"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    panels = [
        ("ct.main (GT)",                   adata.obs["ct.main"].astype(str).values),
        ("ct.sub (GT)",                    adata.obs["ct.sub"].astype(str).values),
        (f"knee picker  k={adata.obs['leiden_bbknn'].nunique()}, r=0.26",
         adata.obs["leiden_bbknn"].astype(str).values),
        (f"CHAMP        k={len(np.unique(champ_labels))}, γ={champ_r:.3f}",
         adata.obs["leiden_bbknn_champ"].astype(str).values),
    ]

    for ax, (title, lbl) in zip(axes.flat, panels):
        cats = sorted(np.unique(lbl), key=lambda s: (len(s), s))
        cmap = plt.colormaps.get_cmap("tab20").resampled(max(20, len(cats)))
        for i, c in enumerate(cats):
            mask = lbl == c
            ax.scatter(
                X[mask, 0], X[mask, 1],
                s=0.6, c=[cmap(i % cmap.N)], label=c,
                rasterized=True, linewidths=0,
            )
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        if len(cats) <= 25:
            ax.legend(
                markerscale=8, fontsize=7, loc="best",
                ncol=2 if len(cats) > 12 else 1, framealpha=0.7,
            )

    plt.suptitle(
        f"222k atlas — CHAMP vs knee picker on bbknn graph\n"
        f"(both ran on identical kNN connectivities; X_umap_bbknn shared)",
        fontsize=14, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))

    out = Path("benchmarks/out/champ_vs_knee_222k.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

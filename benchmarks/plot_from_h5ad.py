"""Post-hoc plotter — re-emit every diagnostic plot from a saved pipeline
output ``.h5ad``.

Useful when a smoke run predated the in-pipeline ``plot_dir`` wiring
(or when you want to re-plot a finished run with different point
sizes / color keys).

Reads ``uns['scib_*']``, ``uns['silhouette_curve_*']``,
``uns['rogue_per_cluster_*']``, ``obs['leiden_*']``, ``obsm['X_umap_*']``.

Usage::

    python benchmarks/plot_from_h5ad.py benchmarks/out/smoke_222k_result.h5ad \\
        --plot-dir benchmarks/out/smoke_222k_plots                           \\
        --label-key ct.main
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5ad", help="path to pipeline-output AnnData")
    parser.add_argument(
        "--plot-dir", default=None,
        help="output dir (default: <h5ad_stem>_plots/ next to the input)",
    )
    parser.add_argument(
        "--label-key", default=None,
        help="ground-truth label column in obs (e.g., ct.main) — overlays on UMAP",
    )
    parser.add_argument(
        "--leiden-target-n", nargs=2, type=int, default=[3, 10],
        help="eligible k-band for the silhouette curve plot",
    )
    args = parser.parse_args()

    import anndata as ad
    from fast_auto_scrna.config import INTEGRATION_METHODS
    from fast_auto_scrna.plotting import emit_route_plots

    in_path = Path(args.h5ad)
    plot_dir = Path(args.plot_dir) if args.plot_dir else in_path.parent / f"{in_path.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {in_path}")
    adata = ad.read_h5ad(in_path)
    print(f"       {adata.n_obs} × {adata.n_vars}")

    # Discover which routes produced outputs.
    methods = tuple(m for m in INTEGRATION_METHODS if f"X_umap_{m}" in adata.obsm)
    if not methods:
        raise SystemExit("no per-route UMAP embeddings found — is this a v2 pipeline output?")
    print(f"[routes] found: {methods}")

    cfg_shim = SimpleNamespace(
        label_key=args.label_key,
        leiden_target_n=tuple(args.leiden_target_n),
    )

    for method in methods:
        print(f"\n── plotting {method} ──")
        written = emit_route_plots(adata, method, plot_dir, cfg_shim)
        for p in written:
            print(f"  {p}")

    # Multi-route comparison plots if we have >1 route.
    if len(methods) > 1:
        from fast_auto_scrna.plotting import (
            compare_integration_plot, compare_scib_heatmap,
            compare_rogue_per_cluster,
        )
        print(f"\n── cross-route comparison ──")
        try:
            p = compare_integration_plot(
                adata, plot_dir / "integration_comparison.png",
                label_key=args.label_key or "_batch",
                methods=methods,
            )
            print(f"  {p}")
        except Exception as e:
            print(f"  integration_comparison failed: {type(e).__name__}: {e}")
        try:
            p = compare_scib_heatmap(
                adata, plot_dir / "scib_heatmap.png", methods=methods,
            )
            print(f"  {p}")
        except Exception as e:
            print(f"  scib_heatmap failed: {type(e).__name__}: {e}")
        try:
            p = compare_rogue_per_cluster(
                adata, plot_dir / "rogue_comparison.png", methods=methods,
            )
            print(f"  {p}")
        except Exception as e:
            print(f"  rogue_comparison failed: {type(e).__name__}: {e}")

    print(f"\n[done] plots in {plot_dir}")


if __name__ == "__main__":
    main()

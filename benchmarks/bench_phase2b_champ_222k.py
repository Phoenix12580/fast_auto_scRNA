"""Validate v2-P12 CHAMP resolution picker against the v2-P10
single-stage knee baseline on the 222k bbknn graph.

CHAMP (Weir et al. 2017) finds the upper convex hull of partitions in
the (b, a) modularity-coefficient plane and picks the hull partition
with the widest *admissible γ-range*. Deterministic + modularity-
principled — no fragile knee detector.

Acceptance:
  * picked γ within ±0.10 of v2-P10 baseline (r=0.26)
    (looser than knee benches because CHAMP optimises a different
    criterion; the test is whether CHAMP lands in the same biological
    "ballpark" k=10..14 region, not whether it bit-matches the heuristic.)
  * n_clusters within ±3 of baseline (k=12)
  * wall < 25% of baseline (1714.9 s)

Usage:
  python benchmarks/bench_phase2b_champ_222k.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _run_champ(adata, *, method: str = "bbknn", **kw):
    from fast_auto_scrna.cluster.resolution import auto_resolution
    from fast_auto_scrna.config import PipelineConfig

    cfg = PipelineConfig(
        input_h5ad="<cached>",
        resolution_optimizer="champ",
        leiden_n_iterations=2,
        max_leiden_workers=None,
        leiden_worker_priority="below_normal",
        **kw,
    )

    conn = adata.obsp[f"{method}_connectivities"]
    t0 = time.perf_counter()
    labels, chosen_res = auto_resolution(adata, method, conn, cfg)
    wall = time.perf_counter() - t0
    n_clusters = int(len(np.unique(labels)))
    return chosen_res, n_clusters, wall, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="benchmarks/out/smoke_222k_v2p10.h5ad")
    ap.add_argument("--method", default="bbknn")
    ap.add_argument("--n-partitions", type=int, default=30)
    ap.add_argument("--gamma-max", type=float, default=1.50)
    ap.add_argument("--width-metric", default="log",
                    choices=["log", "linear", "relative"])
    ap.add_argument("--modularity", default="newman",
                    choices=["newman", "cpm"])
    args = ap.parse_args()

    print(f"[load] {args.h5ad}")
    import anndata as ad
    t0 = time.perf_counter()
    adata = ad.read_h5ad(args.h5ad)
    print(f"  shape {adata.shape}  ({time.perf_counter() - t0:.1f}s)")

    baseline_r = float(adata.uns.get("leiden_bbknn_resolution", float("nan")))
    baseline_k = int(adata.obs["leiden_bbknn"].nunique())
    print(f"\n[baseline] v2-P10 single-stage knee (150-pt sweep):")
    print(f"  picked r        = {baseline_r:.2f}")
    print(f"  n_clusters      = {baseline_k}")
    print(f"  wall (from log) = 1714.9 s (28.6 min)")

    print(f"\n[v2-P12 CHAMP] {args.n_partitions} partitions "
          f"∈ [0.05, {args.gamma_max}], {args.modularity} modularity, "
          f"{args.width_metric} width")
    print("=" * 72)
    ch_r, ch_k, ch_wall, _ = _run_champ(
        adata,
        method=args.method,
        champ_n_partitions=args.n_partitions,
        champ_gamma_max=args.gamma_max,
        champ_modularity=args.modularity,
        champ_width_metric=args.width_metric,
    )

    print("\n" + "=" * 72)
    print("Comparison")
    print("=" * 72)
    print(f"  {'mode':28s} {'picked':>10s} {'k':>5s} {'wall':>10s}")
    print(f"  {'baseline knee 150-pt':28s} {baseline_r:10.2f} {baseline_k:5d}    1714.9 s")
    print(f"  {'CHAMP ' + str(args.n_partitions) + '-part':28s} {ch_r:10.3f} {ch_k:5d}  {ch_wall:8.1f} s")
    speedup = 1714.9 / ch_wall if ch_wall > 0 else float("inf")
    saved = 1714.9 - ch_wall
    print(f"\n  speedup vs baseline : {speedup:.2f}×")
    print(f"  wall saved          : {saved:.1f} s ({saved/60:.1f} min)")
    print(f"  Δ picked            : {abs(ch_r - baseline_r):.2f}")
    print(f"  Δ n_clusters        : {abs(ch_k - baseline_k)}")

    fail = []
    if abs(ch_r - baseline_r) > 0.10:
        fail.append(f"picked γ drift {ch_r:.3f} vs baseline {baseline_r:.2f} > 0.10")
    if abs(ch_k - baseline_k) > 3:
        fail.append(f"n_clusters drift {ch_k} vs baseline {baseline_k} > 3")
    if ch_wall > 1714.9 * 0.25:
        fail.append(f"CHAMP wall {ch_wall:.0f}s > 25% of baseline 1714.9s")

    if fail:
        print(f"\n[FAIL]")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\n[OK] CHAMP matches baseline within tolerance + meaningful speedup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

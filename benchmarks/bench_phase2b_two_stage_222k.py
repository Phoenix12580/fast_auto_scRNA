"""Validate v2-P12 two-stage knee picker against the v2-P10 single-stage
baseline on the 222k bbknn graph.

Reuses the cached `smoke_222k_v2p10.h5ad` (which has bbknn_connectivities
in obsp from the original baseline run). Re-runs ONLY auto_resolution
in both single-stage and two-stage modes, comparing picked_r,
n_clusters, wall time, and downstream ROGUE/SCCAF on the picked labels.

Acceptance:
  * picked_r within ±0.05 of v2-P10 baseline (r=0.40)
  * n_clusters at picked_r within ±2 of baseline (12)
  * two-stage wall < 25% of single-stage wall

Usage:
  python benchmarks/bench_phase2b_two_stage_222k.py \
      --h5ad benchmarks/out/smoke_222k_v2p10.h5ad
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _run_auto_res(adata, *, two_stage: bool, method: str = "bbknn"):
    """Re-run auto_resolution on the cached graph in the chosen mode."""
    from fast_auto_scrna.cluster.resolution import auto_resolution
    from fast_auto_scrna.config import PipelineConfig

    cfg = PipelineConfig(
        input_h5ad="<cached>",
        resolution_optimizer="knee",
        knee_offset_steps=3,
        knee_detector="first_plateau",
        knee_two_stage=two_stage,
        knee_fine_step=0.01,
        knee_fine_half_width=0.05,
        leiden_n_iterations=2,
        # match v2-P10 baseline worker config
        max_leiden_workers=None,
        leiden_worker_priority="below_normal",
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
    args = ap.parse_args()

    print(f"[load] {args.h5ad}")
    import anndata as ad
    t0 = time.perf_counter()
    adata = ad.read_h5ad(args.h5ad)
    print(f"  shape {adata.shape}  ({time.perf_counter() - t0:.1f}s)")

    # Baseline reference (from v2-P10 stored result)
    baseline_r = float(adata.uns.get("leiden_bbknn_resolution", float("nan")))
    baseline_k = int(adata.obs["leiden_bbknn"].nunique())
    print(f"\n[baseline] v2-P10 single-stage 150-pt:")
    print(f"  picked r        = {baseline_r:.2f}")
    print(f"  n_clusters      = {baseline_k}")
    print(f"  wall (from log) = 1714.9 s (28.6 min)")

    print("\n" + "=" * 72)
    print(f"Two-stage knee picker (v2-P12 default)")
    print("=" * 72)
    ts_r, ts_k, ts_wall, _ = _run_auto_res(adata, two_stage=True, method=args.method)

    print("\n" + "=" * 72)
    print("Comparison")
    print("=" * 72)
    print(f"  {'mode':22s} {'picked_r':>10s} {'k':>5s} {'wall':>10s}")
    print(f"  {'baseline (single)':22s} {baseline_r:10.2f} {baseline_k:5d}    1714.9 s")
    print(f"  {'v2-P12 two-stage':22s} {ts_r:10.2f} {ts_k:5d}  {ts_wall:8.1f} s")
    speedup = 1714.9 / ts_wall if ts_wall > 0 else float("inf")
    print(f"\n  speedup vs baseline : {speedup:.2f}×")
    print(f"  wall saved          : {(1714.9 - ts_wall):.1f} s ({(1714.9 - ts_wall)/60:.1f} min)")
    print(f"  Δ picked_r          : {abs(ts_r - baseline_r):.2f}")
    print(f"  Δ n_clusters        : {abs(ts_k - baseline_k)}")

    fail = []
    if abs(ts_r - baseline_r) > 0.05:
        fail.append(f"picked_r drift {ts_r:.2f} vs baseline {baseline_r:.2f} > 0.05")
    if abs(ts_k - baseline_k) > 2:
        fail.append(f"n_clusters drift {ts_k} vs baseline {baseline_k} > 2")
    if ts_wall > 1714.9 * 0.25:
        fail.append(f"two-stage wall {ts_wall:.0f}s > 25% of baseline 1714.9s")

    if fail:
        print(f"\n[FAIL]")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\n[OK] two-stage matches baseline within tolerance + meaningful speedup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

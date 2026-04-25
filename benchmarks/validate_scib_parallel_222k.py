"""Validate v2-P11 parallel Phase 2a scIB on the 222k v2-P10 baseline.

Re-runs ONLY the Phase 2a scIB block on cached artifacts from the
`smoke_222k_v2p10.h5ad` baseline. Avoids the ~80 min of re-running the
full pipeline by reusing the saved kNN + embedding + labels.

Asserts:
  * parallel scIB == sequential scIB (|delta| < 1e-4 per metric)
  * parallel/sequential scIB == stored baseline `scib_<route>` (same tol)

Reports wall:
  * sequential 4-route Phase 2a wall
  * parallel 4-route Phase 2a wall + speedup

Usage:
  python benchmarks/validate_scib_parallel_222k.py \
      --h5ad benchmarks/out/smoke_222k_v2p10.h5ad \
      --label-key ct.main
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def _build_artifacts(adata, methods):
    """Pull `(knn dict, embed)` for every route out of the cached h5ad."""
    arts = {}
    for m in methods:
        knn = adata.uns[f"{m}_knn"]
        knn_dict = {
            "indices": np.asarray(knn["indices"], dtype=np.uint32),
            "distances": np.asarray(knn["distances"], dtype=np.float32),
        }
        embed = adata.obsm[f"X_pca_{m}"]
        arts[m] = {"knn": knn_dict, "embed": np.ascontiguousarray(embed, dtype=np.float32)}
    return arts


def _run_phase2a(adata, methods, arts, *, parallel: bool, label_key: str):
    """Invoke runner orchestrator with the chosen scheduling mode."""
    from fast_auto_scrna.runner import _phase2a_scib_all_routes
    from fast_auto_scrna.config import PipelineConfig

    cfg = PipelineConfig(
        input_h5ad="<cached>",
        label_key=label_key,
        compute_silhouette=True,
        compute_kbet=False,
        scib_parallel=parallel,
        scib_max_workers=None,
    )
    route_timings = {m: {} for m in methods}

    # _phase2a_scib_all_routes writes scib_{m} into adata.uns. Strip stored
    # baseline first so we measure the freshly-computed values.
    fresh = adata.copy()
    for m in methods:
        fresh.uns.pop(f"scib_{m}", None)

    t0 = time.perf_counter()
    _phase2a_scib_all_routes(fresh, methods, arts, cfg, route_timings)
    wall = time.perf_counter() - t0

    out = {m: dict(fresh.uns[f"scib_{m}"]) for m in methods}
    per_route = {m: route_timings[m].get("metrics", float("nan")) for m in methods}
    return out, wall, per_route


# Phase 2a outputs only — rogue_*/sccaf are Phase 2b homogeneity metrics
# that the runner stores into the same scib_<route> dict downstream.
PHASE2A_KEYS = {
    "ilisi", "clisi", "graph_connectivity", "kbet_acceptance",
    "label_silhouette", "batch_silhouette", "isolated_label", "mean",
}


def _compare(label, ref, got, tol=1e-4):
    """Print + check metric-by-metric absolute differences.

    Only checks keys produced by Phase 2a — Phase 2b outputs (rogue/sccaf)
    in the stored baseline are not reproducible from a Phase-2a-only re-run.
    The Phase 2a 'mean' includes only Phase 2a metrics, so it's comparable.
    """
    bad = []
    print(f"\n=== {label} ===")
    print(f"  {'metric':22s} {'ref':>12s} {'got':>12s} {'|Δ|':>10s}")
    for k in sorted(ref.keys()):
        if k not in PHASE2A_KEYS:
            continue
        if not isinstance(ref[k], (int, float)):
            continue
        v_ref = float(ref[k])
        v_got = float(got.get(k, float("nan")))
        delta = abs(v_got - v_ref)
        ok = (
            delta < tol
            or (np.isnan(v_ref) and np.isnan(v_got))
        )
        marker = "" if ok else "  ✗"
        if not ok:
            bad.append((k, v_ref, v_got, delta))
        print(f"  {k:22s} {v_ref:12.6f} {v_got:12.6f} {delta:10.2e}{marker}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="benchmarks/out/smoke_222k_v2p10.h5ad")
    ap.add_argument("--label-key", default="ct.main",
                    help="Cell-type column in obs (default ct.main)")
    ap.add_argument("--routes", nargs="+",
                    default=["bbknn", "harmony", "fastmnn", "scvi"])
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    print(f"[load] {args.h5ad}")
    import anndata as ad
    t0 = time.perf_counter()
    adata = ad.read_h5ad(args.h5ad)
    print(f"  shape {adata.shape}  ({time.perf_counter() - t0:.1f}s)")

    methods = tuple(args.routes)
    arts = _build_artifacts(adata, methods)

    # Stored baseline values for reference
    baseline = {m: dict(adata.uns[f"scib_{m}"]) for m in methods}

    print("\n" + "=" * 72)
    print("Phase 2a scIB — sequential")
    print("=" * 72)
    seq_out, seq_wall, seq_per = _run_phase2a(
        adata, methods, arts, parallel=False, label_key=args.label_key,
    )

    print("\n" + "=" * 72)
    print("Phase 2a scIB — parallel (v2-P11)")
    print("=" * 72)
    par_out, par_wall, par_per = _run_phase2a(
        adata, methods, arts, parallel=True, label_key=args.label_key,
    )

    print("\n" + "=" * 72)
    print(f"Numerical equivalence check (tol={args.tol:g})")
    print("=" * 72)
    failures = []
    for m in methods:
        bad_seq_vs_baseline = _compare(
            f"{m}: sequential vs stored baseline", baseline[m], seq_out[m], args.tol,
        )
        bad_par_vs_seq = _compare(
            f"{m}: parallel vs sequential",
            seq_out[m], par_out[m], args.tol,
        )
        if bad_seq_vs_baseline:
            failures.append((m, "seq vs baseline", bad_seq_vs_baseline))
        if bad_par_vs_seq:
            failures.append((m, "par vs seq", bad_par_vs_seq))

    print("\n" + "=" * 72)
    print("Wall time")
    print("=" * 72)
    print(f"  sequential 4-route wall : {seq_wall:7.1f} s ({seq_wall/60:.1f} min)")
    print(f"  parallel 4-route wall   : {par_wall:7.1f} s ({par_wall/60:.1f} min)")
    if par_wall > 0:
        print(f"  speedup                 : {seq_wall/par_wall:5.2f}×")
    print(f"\n  per-route worker time:")
    for m in methods:
        print(f"    {m:10s} seq={seq_per[m]:6.1f}s   par={par_per[m]:6.1f}s")

    if failures:
        print("\n[FAIL] numerical mismatches:")
        for m, kind, bad in failures:
            for k, ref, got, d in bad:
                print(f"  {m}/{kind}: {k} ref={ref!r} got={got!r} |Δ|={d:.3e}")
        return 1
    print("\n[OK] all metrics within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

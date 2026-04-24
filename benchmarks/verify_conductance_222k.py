"""End-to-end verify: conductance optimizer on 222k bbknn graph.

Runs ``optimize_resolution_conductance`` against the real 222k atlas,
prints the curve, compares picked resolution to ground-truth ct.main ARI.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import time
import numpy as np
import anndata as ad
from sklearn.metrics import adjusted_rand_score

from fast_auto_scrna.cluster.resolution import (
    optimize_resolution_conductance,
    pick_best_resolution,
)

H5AD = "benchmarks/out/smoke_222k_all_gs3.h5ad"


def main():
    a = ad.read_h5ad(H5AD)
    print(f"loaded: {a.n_obs} cells")
    # Drop pre-existing leiden cols so optimizer writes fresh.
    for c in [c for c in a.obs.columns if c.startswith("leiden_bbknn_r")]:
        del a.obs[c]

    t0 = time.perf_counter()
    curve = optimize_resolution_conductance(
        a, method="bbknn",
        resolutions=[0.05, 0.1, 0.2, 0.3, 0.5],
        seed=0, leiden_n_iterations=2, verbose=True,
    )
    dt = time.perf_counter() - t0
    print()
    print(f"optimize_resolution_conductance wall: {dt:.2f}s")
    print(curve.to_string(index=False))

    best_r = pick_best_resolution(
        curve, k_lo=3, k_hi=10,
        metric="conductance", direction="min",
    )
    best_row = curve[curve["resolution"] == best_r].iloc[0]
    print(f"\nconductance picker (k∈[3,10]): r={best_r:.2f} "
          f"(k={int(best_row['n_clusters'])}, cond={best_row['conductance']:.4f})")

    manual = a.obs["ct.main"].astype(str).to_numpy()
    print(f"\nARI vs ct.main (n_classes={len(set(manual))}):")
    for _, row in curve.iterrows():
        key = f"leiden_bbknn_r{row['resolution']:.2f}"
        lbl = a.obs[key].astype(int).to_numpy()
        ari = adjusted_rand_score(manual, lbl)
        star = " ← picked" if row["resolution"] == best_r else ""
        print(f"  r={row['resolution']:.2f} k={int(row['n_clusters']):3d}  "
              f"ARI={ari:.4f}{star}")


if __name__ == "__main__":
    main()

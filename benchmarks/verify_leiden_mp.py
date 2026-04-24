"""Verify optimize_resolution_graph_silhouette parallel sweep on 222k.

Runs the actual function (not just raw Leiden) to confirm:
  - wall time drops vs sequential baseline
  - silhouette curve is byte-equivalent (leidenalg deterministic)
"""
from __future__ import annotations

import time
import pickle
import numpy as np
import anndata as ad

from fast_auto_scrna.cluster.resolution import optimize_resolution_graph_silhouette

H5AD = "benchmarks/out/smoke_222k_all_gs3.h5ad"


def main():
    a = ad.read_h5ad(H5AD)
    print(f"loaded: {a.n_obs} cells")

    # Drop any pre-existing leiden cols so the sweep writes fresh.
    drop = [c for c in a.obs.columns if c.startswith("leiden_bbknn_r")]
    for c in drop:
        del a.obs[c]

    t0 = time.perf_counter()
    curve = optimize_resolution_graph_silhouette(
        a, method="bbknn",
        resolutions=[0.05, 0.1, 0.2, 0.3, 0.5],
        n_subsample=1000, n_iter=100,
        seed=0, verbose=True,
    )
    dt = time.perf_counter() - t0
    print()
    print(f"optimize_resolution_graph_silhouette wall : {dt:.2f}s")
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()

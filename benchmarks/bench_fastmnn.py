"""Standalone microbench for fastMNN on 222k atlas.

Loads X_pca + _batch from the latest 222k h5ad (via the h5py-only
loader from diagnose_asw.py) and runs fastmnn with default params.
Reports wall, n_pairs per merge, output sanity.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diagnose_asw import _load_obs_obsm_only

from fast_auto_scrna.integration import fastmnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="benchmarks/out/smoke_222k_all_v2p9.h5ad")
    ap.add_argument("--batch-key", default="_batch")
    ap.add_argument("--n-neighbors", type=int, default=20)
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    args = ap.parse_args()

    print(f"loading {args.h5ad} ...")
    t0 = time.perf_counter()
    adata = _load_obs_obsm_only(args.h5ad)
    print(f"  {adata.n_obs} cells in {time.perf_counter() - t0:.1f}s")

    if "X_pca" not in adata.obsm:
        print("X_pca missing — falling back to X_pca_none / X_pca_bbknn")
        for k in ("X_pca_none", "X_pca_bbknn"):
            if k in adata.obsm:
                pca = adata.obsm[k]
                break
        else:
            print(f"no PCA in obsm; have {list(adata.obsm.keys())}")
            return 1
    else:
        pca = adata.obsm["X_pca"]

    batch_arr = adata.obs[args.batch_key]
    if hasattr(batch_arr, "cat"):
        batch_codes = batch_arr.cat.codes.to_numpy()
    else:
        batch_codes = np.asarray(batch_arr)

    print(f"\nPCA shape: {pca.shape}, dtype={pca.dtype}")
    uniq, counts = np.unique(batch_codes, return_counts=True)
    print(f"batches: {len(uniq)}")
    for u, c in sorted(zip(uniq, counts), key=lambda kv: -kv[1]):
        print(f"  batch {u:>3}: n={c}")

    print(f"\nrunning fastmnn(k={args.n_neighbors}, sigma_scale={args.sigma_scale}) ...")
    t0 = time.perf_counter()
    result = fastmnn(
        pca.astype(np.float32, copy=False), batch_codes,
        n_neighbors=args.n_neighbors,
        sigma_scale=args.sigma_scale,
    )
    wall = time.perf_counter() - t0

    print(f"\n=== RESULT ===")
    print(f"wall:                 {wall:.1f}s ({wall / 60:.1f} min)")
    print(f"merge_order:          {result['merge_order']}")
    print(f"n_pairs_per_merge:    {result['n_pairs_per_merge']}")
    print(f"skipped_batches:      {result['skipped_batches']}")
    print(f"corrected.shape:      {result['corrected'].shape}")
    print(f"corrected.dtype:      {result['corrected'].dtype}")
    cor = result["corrected"]
    print(f"corrected stats:      "
          f"mean_abs={np.mean(np.abs(cor)):.4f}  "
          f"min={cor.min():.4f}  max={cor.max():.4f}")
    diff_norm = np.linalg.norm(cor - pca / (np.linalg.norm(pca, axis=1, keepdims=True) + 1e-10).astype(np.float32))
    print(f"||corrected - input_normalized|| (Frobenius): {diff_norm:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

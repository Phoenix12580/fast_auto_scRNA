"""Standalone scVI benchmark on the 222k atlas (GPU vs CPU comparison).

Loads ``data/StepF.All_Cells.h5ad``, runs minimal preprocess (lognorm +
seurat HVG = 2k genes), then trains SCVI with full default params
(n_latent=30, max_epochs=200) and reports wall + latent shape.

Use ``--max-epochs N`` to cap training for quick checks.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/StepF.All_Cells.h5ad")
    ap.add_argument("--batch-key", default="data.sets")
    ap.add_argument("--n-latent", type=int, default=30)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--accelerator", default="auto",
                    choices=["auto", "gpu", "cpu"])
    args = ap.parse_args()

    import anndata as ad
    import scanpy as sc

    print(f"loading {args.input} ...")
    t_load = time.perf_counter()
    adata = ad.read_h5ad(args.input)
    print(f"  loaded {adata.n_obs} cells × {adata.n_vars} genes "
          f"in {time.perf_counter() - t_load:.1f}s")
    print(f"  obs columns: {list(adata.obs.columns)[:10]}")

    # Keep raw counts in layers (scvi requirement)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Minimal preprocess: lognorm + HVG (seurat flavor matches the 222k
    # production path; seurat_v3 segfaults on 10-batch x 222k)
    print("\nlognorm + HVG selection (seurat flavor)...")
    t_pp = time.perf_counter()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=args.n_hvg, flavor="seurat",
        batch_key=args.batch_key,
    )
    print(f"  HVGs: {int(adata.var['highly_variable'].sum())} / {adata.n_vars} "
          f"in {time.perf_counter() - t_pp:.1f}s")

    # Verify GPU availability
    import torch
    print(f"\ntorch={torch.__version__} cuda_compiled={torch.version.cuda} "
          f"available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # Train scVI
    from fast_auto_scrna.integration import scvi_train

    print(f"\ntraining scVI (n_latent={args.n_latent}, max_epochs={args.max_epochs}, "
          f"accelerator={args.accelerator}) ...")
    t_train = time.perf_counter()
    latent, info = scvi_train(
        adata,
        batch_key=args.batch_key,
        n_latent=args.n_latent,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        enable_progress_bar=True,
    )
    train_wall = time.perf_counter() - t_train

    print(f"\n=== RESULT ===")
    print(f"latent shape:     {latent.shape}")
    print(f"latent dtype:     {latent.dtype}")
    print(f"latent stats:     mean_abs={float(np.mean(np.abs(latent))):.4f}  "
          f"std={float(np.std(latent)):.4f}")
    print(f"train wall:       {train_wall:.1f} s ({train_wall / 60:.1f} min)")
    print(f"trained on:       {info['n_train_cells']} cells × "
          f"{info['n_train_genes']} HVG genes")
    print(f"epochs:           {info['max_epochs']}")
    print(f"accelerator:      {info['accelerator']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

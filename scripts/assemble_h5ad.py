"""Reassemble sparse counts HDF5 + obs CSV + genes/cells → AnnData h5ad.

Paired with convert_qs_to_h5ad.R. Expects the three files that R wrote.

Run:
    wsl bash -c "/mnt/f/NMF_rewrite/scatlas/.venv/bin/python \\
        /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scripts/assemble_h5ad.py"
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata

CONV_DIR = Path("/mnt/f/NMF_rewrite/_stepf_conv")
OUT = Path("/mnt/f/NMF_rewrite/StepF.All_Cells.h5ad")


def main():
    t0 = time.perf_counter()
    with h5py.File(CONV_DIR / "counts_csc.h5", "r") as f:
        data = f["data"][...].astype(np.float32, copy=False)
        indices = f["indices"][...].astype(np.int32, copy=False)
        indptr = f["indptr"][...].astype(np.int32, copy=False)
        shape = tuple(f["shape"][...].astype(int).tolist())
    print(f"[py] HDF5 load: {time.perf_counter() - t0:.1f}s; "
          f"shape genes×cells={shape}, nnz={len(data)}")

    # dgCMatrix from R is CSC with shape (n_genes, n_cells)
    X_gxc = sp.csc_matrix((data, indices, indptr), shape=shape)
    # AnnData wants cells × genes, CSR is faster for row-wise slicing
    X_cxg = X_gxc.T.tocsr()
    print(f"[py] CSC → CSR (cells × genes): {X_cxg.shape}, {X_cxg.nnz} nnz")

    t0 = time.perf_counter()
    obs = pd.read_csv(CONV_DIR / "obs.csv", index_col="cell")
    obs.index = obs.index.astype(str)
    obs.index.name = None
    genes = [ln.strip() for ln in open(CONV_DIR / "genes.txt")]
    cells = [ln.strip() for ln in open(CONV_DIR / "cells.txt")]
    assert len(cells) == X_cxg.shape[0] == len(obs), (
        f"mismatch: {len(cells)} cells / {X_cxg.shape[0]} X rows / {len(obs)} obs rows"
    )
    assert len(genes) == X_cxg.shape[1], f"{len(genes)} genes / {X_cxg.shape[1]} vars"

    var = pd.DataFrame(index=pd.Index(genes, dtype="object"))
    obs.index = pd.Index(cells, dtype="object")

    adata = anndata.AnnData(X=X_cxg, obs=obs, var=var)
    # Preserve counts layer so the pipeline's layers["counts"] lookup works
    adata.layers["counts"] = X_cxg.copy()
    print(f"[py] AnnData built in {time.perf_counter() - t0:.1f}s "
          f"→ {adata.shape}, obs cols: {list(adata.obs.columns)}")

    t0 = time.perf_counter()
    adata.write_h5ad(OUT, compression="gzip")
    sz = OUT.stat().st_size / 1e9
    print(f"[py] wrote {OUT} in {time.perf_counter() - t0:.1f}s ({sz:.2f} GB)")


if __name__ == "__main__":
    main()

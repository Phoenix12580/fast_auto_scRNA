"""Assemble AnnData from the R-side dump → data/pancreas_sub.h5ad."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import anndata as ad

PARTS = Path("data/pancreas_sub_parts")
OUT   = Path("data/pancreas_sub.h5ad")


def main():
    X = sio.mmread(PARTS / "counts.mtx").tocsr()      # genes × cells
    X = X.T.tocsr()                                    # cells × genes
    obs_names = [l.strip() for l in (PARTS / "obs_names.txt").read_text().splitlines()]
    var_names = [l.strip() for l in (PARTS / "var.txt").read_text().splitlines()]
    obs = pd.read_csv(PARTS / "obs.csv", index_col=0)
    obs.index = obs_names
    var = pd.DataFrame(index=var_names)

    # Pipeline expects an 'orig.ident' or similar batch col; pancreas has one
    # value only, so integration is a no-op here. For v2 smoke we'll pass
    # batch_key=None / integration="none".
    a = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    a.write_h5ad(OUT, compression="gzip")
    print(f"wrote {OUT}: {a.n_obs} cells × {a.n_vars} genes")
    print(f"obs cols: {list(a.obs.columns)}")
    print(f"CellType counts:\n{a.obs['CellType'].value_counts()}")
    print(f"SubCellType counts:\n{a.obs['SubCellType'].value_counts()}")


if __name__ == "__main__":
    main()

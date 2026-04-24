"""Per-cell QC filtering. Ported from v1 ``pipeline._qc_filter``."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def qc_filter(adata, cfg):
    """Drop genes in < min_cells cells, cells with < min_genes genes, and
    cells whose MT fraction exceeds max_pct_mt. Returns a new AnnData.
    """
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)

    gene_cells = (X > 0).sum(axis=0).A1 if sp.issparse(X) else (X > 0).sum(axis=0)
    keep_g = gene_cells >= cfg.min_cells

    cell_genes = (X > 0).sum(axis=1).A1 if sp.issparse(X) else (X > 0).sum(axis=1)
    keep_c = cell_genes >= cfg.min_genes

    if 0 < cfg.max_pct_mt < 100:
        mt = np.array(
            [str(v).upper().startswith(cfg.mt_prefix.upper()) for v in adata.var_names]
        )
        if mt.any():
            libs = np.asarray(X.sum(axis=1)).ravel()
            libs_safe = np.where(libs == 0, 1, libs)
            mt_counts = np.asarray(X[:, mt].sum(axis=1)).ravel()
            pct_mt = 100.0 * mt_counts / libs_safe
            keep_c = keep_c & (pct_mt <= cfg.max_pct_mt)

    return adata[keep_c, :][:, keep_g].copy()

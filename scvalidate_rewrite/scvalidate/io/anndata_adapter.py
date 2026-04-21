"""AnnData → internal matrix adapter.

scvalidate works internally with genes × cells raw-count matrices (matching
the R upstream conventions of recall / sc-SHC / ROGUE). AnnData stores
cells × genes, so every entry point transposes once here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def get_counts_matrix(
    adata: AnnData, layer: str | None = None
) -> sp.csr_matrix | np.ndarray:
    """Return raw counts as a genes × cells matrix.

    Parameters
    ----------
    adata
        AnnData object. Expected to have integer-valued raw counts in X
        (or in the named ``layer``). The three upstream R methods all assume
        raw counts; if the user has already normalized they must pass
        ``layer='counts'`` or similar.
    layer
        Optional layer name. If None, uses ``adata.X``.
    """
    X = adata.X if layer is None else adata.layers[layer]
    # AnnData is cells × genes; transpose to genes × cells
    if sp.issparse(X):
        return X.T.tocsr()
    return np.asarray(X).T


def subset_cells(
    counts_gxc: sp.csr_matrix | np.ndarray, cell_idx: Sequence[int]
) -> sp.csr_matrix | np.ndarray:
    """Subset a genes × cells matrix by cell indices."""
    cell_idx = np.asarray(cell_idx, dtype=np.int64)
    if sp.issparse(counts_gxc):
        return counts_gxc[:, cell_idx]
    return counts_gxc[:, cell_idx]


def ensure_dense(mat) -> np.ndarray:
    """Densify a matrix once if sparse; pass through otherwise."""
    if sp.issparse(mat):
        return np.asarray(mat.todense())
    return np.asarray(mat)

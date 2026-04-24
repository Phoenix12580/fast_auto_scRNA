"""Stage 02 — normalize + log1p. Ported from v1 ``pipeline._lognorm``.

Kept pure-numpy / scipy.sparse for now. A Rust kernel is planned under
OOM-1 (chunked preprocess) so this stays on a known interface.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def lognorm(X, target_sum: float = 1e4):
    """Library-size-normalize each cell to ``target_sum`` then log1p.

    Returns a float32 CSR matrix matching the input shape.
    """
    X = X if sp.issparse(X) else sp.csr_matrix(X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((target_sum / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    return Xn.tocsr()

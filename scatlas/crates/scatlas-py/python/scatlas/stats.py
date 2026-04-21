"""Robust statistics — thin wrappers around the Rust kernels.

The Rust layer (`scatlas._scatlas_native.stats`) returns flat buffers; this
module reshapes them into the shape Python callers expect, and provides the
ergonomic defaults (platform → k for ROGUE, etc.).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from scatlas._scatlas_native.stats import (
    calculate_rogue as _calculate_rogue_native,
    entropy_table as _entropy_table_native,
    entropy_table_csr as _entropy_table_csr_native,
    knockoff_threshold_offset1 as _knockoff_native,
    wilcoxon_ranksum_matrix as _wilcoxon_native,
)

__all__ = [
    "wilcoxon_ranksum_matrix",
    "knockoff_threshold_offset1",
    "entropy_table",
    "calculate_rogue",
]


def wilcoxon_ranksum_matrix(
    log_counts: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> np.ndarray:
    """Per-row Wilcoxon rank-sum p-values (rows = genes, cols = cells).

    Matches ``scvalidate.recall_py.core._wilcoxon_per_gene`` bit-for-bit.
    Accepts float32 or float64 ``log_counts``; masks must be bool.
    """
    return _wilcoxon_native(log_counts, mask1.astype(bool), mask2.astype(bool))


def knockoff_threshold_offset1(w: np.ndarray, fdr: float) -> float:
    """Barber-Candès knockoff threshold (offset=1)."""
    w64 = np.ascontiguousarray(w, dtype=np.float64)
    return _knockoff_native(w64, fdr)


def entropy_table(expr, r: float = 1.0) -> np.ndarray:
    """Per-gene ROGUE entropy table; returns ``(n_genes, 2)`` of
    ``[log(mean+r), mean(log(x+1))]``.

    Accepts dense float32/float64 ndarray (genes × cells) or a
    ``scipy.sparse.csr_matrix``. For sparse input, data buffer dtype must be
    float32 or float64; indices/indptr may be any integer type.
    """
    try:
        import scipy.sparse as sp  # type: ignore
    except ImportError:
        sp = None

    if sp is not None and sp.issparse(expr):
        csr = expr.tocsr()
        n_genes, n_cells = csr.shape
        indptr = np.asarray(csr.indptr, dtype=np.int64)
        data = np.ascontiguousarray(csr.data)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float64)
        flat = _entropy_table_csr_native(indptr, data, int(n_cells), float(r))
    else:
        arr = np.asarray(expr)
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float64)
        flat = _entropy_table_native(arr, float(r))

    return flat.reshape(-1, 2)


def calculate_rogue(
    ds: np.ndarray,
    p_adj: np.ndarray,
    p_value: np.ndarray,
    platform: Literal["UMI", "full-length"] | None = None,
    k: float | None = None,
    cutoff: float = 0.05,
) -> float:
    """Single ROGUE score from per-gene ``(ds, p_adj, p_value)``.

    ``k`` falls back to 45 (UMI) or 500 (full-length) when ``platform`` is
    given; otherwise one of ``k`` or ``platform`` is required.
    """
    if k is None:
        if platform == "UMI":
            k = 45.0
        elif platform == "full-length":
            k = 500.0
        else:
            raise ValueError(
                "Must provide `k` or `platform` in {'UMI', 'full-length'}"
            )
    return _calculate_rogue_native(
        np.ascontiguousarray(ds, dtype=np.float64),
        np.ascontiguousarray(p_adj, dtype=np.float64),
        np.ascontiguousarray(p_value, dtype=np.float64),
        float(cutoff),
        float(k),
    )

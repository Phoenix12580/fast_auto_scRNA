"""BBKNN: batch-balanced k-NN over a PCA embedding.

Ported from v1 ``scatlas.ext.bbknn_kneighbors`` / ``bbknn`` at V2-P2.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


def bbknn_kneighbors(
    pca: np.ndarray,
    batch_labels: np.ndarray,
    neighbors_within_batch: int = 3,
    backend: str = "auto",
    ef_search: int | None = None,
    auto_threshold: int = 5000,
) -> dict[str, Any]:
    """Batch-balanced k-NN over a PCA embedding.

    Returns dict with ``indices``, ``distances``, ``batches``,
    ``batch_code_map``, ``backend_used``.
    """
    from fast_auto_scrna._native import bbknn as _native_bbknn

    pca_f32 = np.ascontiguousarray(pca, dtype=np.float32)
    batch_arr = np.asarray(batch_labels)

    code_map: dict[int, Any] = {}
    if batch_arr.dtype.kind in {"i", "u"}:
        batch_int32 = batch_arr.astype(np.int32)
    else:
        uniq, codes = np.unique(batch_arr, return_inverse=True)
        batch_int32 = codes.astype(np.int32)
        code_map = {int(i): uniq[i] for i in range(len(uniq))}

    if backend not in {"brute", "hnsw", "auto"}:
        raise ValueError(
            f"backend must be 'brute' / 'hnsw' / 'auto', got {backend!r}"
        )

    backend_used = backend
    if backend == "auto":
        _, counts = np.unique(batch_int32, return_counts=True)
        backend_used = "hnsw" if counts.max() >= auto_threshold else "brute"

    idx, dist, batches = _native_bbknn.bbknn_kneighbors(
        pca_f32, batch_int32,
        int(neighbors_within_batch), backend, ef_search,
        int(auto_threshold),
    )
    return {
        "indices": idx,
        "distances": dist,
        "batches": batches,
        "batch_code_map": code_map,
        "backend_used": backend_used,
    }


def _distances_to_csr(
    indices: np.ndarray, distances: np.ndarray, n_cells: int,
) -> sp.csr_matrix:
    """Pack the ``(n_cells, k_total)`` kNN output into a sparse distance matrix,
    dropping ``u32::MAX`` padding slots from small batches."""
    valid = indices != np.iinfo(np.uint32).max
    rows = np.broadcast_to(
        np.arange(n_cells, dtype=np.int64)[:, None], indices.shape
    )[valid]
    cols = indices[valid].astype(np.int64)
    data = distances[valid].astype(np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))


def bbknn(
    adata,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
    neighbors_within_batch: int = 3,
    with_connectivities: bool = True,
    key_added: str | None = None,
) -> dict[str, Any]:
    """scanpy-compatible BBKNN entry point.

    Writes ``adata.obsp[f'{key}_distances']`` (always) and
    ``adata.obsp[f'{key}_connectivities']`` (when ``with_connectivities``).
    """
    from fast_auto_scrna.neighbors.knn_fuzzy import fuzzy_connectivities

    if use_rep not in adata.obsm:
        raise KeyError(
            f"{use_rep!r} not in adata.obsm — run PCA first or pass use_rep"
        )
    if batch_key not in adata.obs:
        raise KeyError(f"{batch_key!r} not in adata.obs")

    result = bbknn_kneighbors(
        adata.obsm[use_rep],
        adata.obs[batch_key].to_numpy(),
        neighbors_within_batch=neighbors_within_batch,
    )

    n_cells = adata.n_obs
    dist_csr = _distances_to_csr(result["indices"], result["distances"], n_cells)
    conn_csr = None
    if with_connectivities:
        conn_csr = fuzzy_connectivities(
            result["indices"], result["distances"], n_cells,
        )

    key = key_added or "bbknn"
    adata.obsp[f"{key}_distances"] = dist_csr
    if conn_csr is not None:
        adata.obsp[f"{key}_connectivities"] = conn_csr
    adata.uns[key] = {
        "connectivities_key": f"{key}_connectivities" if conn_csr is not None else None,
        "distances_key": f"{key}_distances",
        "params": {
            "method": "bbknn",
            "use_rep": use_rep,
            "batch_key": batch_key,
            "neighbors_within_batch": neighbors_within_batch,
            "backend": result.get("backend_used", "fast_auto_scrna"),
        },
    }

    return {
        "distances": dist_csr,
        "connectivities": conn_csr,
        "batches": result["batches"],
        "batch_code_map": result["batch_code_map"],
    }

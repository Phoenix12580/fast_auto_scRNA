"""kNN + fuzzy_simplicial_set connectivities.

``fuzzy_connectivities`` is the UMAP-style symmetric fuzzy set built from
any kNN output. ``knn_and_fuzzy`` bundles a BBKNN/none kNN call + fuzzy
CSR into one helper — used by the per-route Phase-1 driver.

Ported from v1 ``scatlas.ext._fuzzy_connectivities`` and
``pipeline._knn_and_fuzzy`` at V2-P2.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


def fuzzy_connectivities(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_cells: int,
) -> sp.csr_matrix:
    """UMAP ``fuzzy_simplicial_set`` from raw kNN arrays.

    BBKNN pads small batches with ``u32::MAX`` / ``inf``; the Rust kernel
    skips those slots internally. Falls back to ``umap-learn``, then a
    per-cell Gaussian kernel, if the Rust binding is unavailable.
    """
    MAX = np.iinfo(np.uint32).max

    try:
        from fast_auto_scrna._native import fuzzy as _native_fuzzy

        idx_u32 = np.ascontiguousarray(knn_indices, dtype=np.uint32)
        dst_f32 = np.ascontiguousarray(knn_distances, dtype=np.float32)
        k_total = idx_u32.shape[1]
        indptr, indices, data = _native_fuzzy.fuzzy_simplicial_set(
            idx_u32, dst_f32,
            k=k_total, n_iter=64,
            set_op_mix_ratio=1.0, local_connectivity=1.0,
        )
        return sp.csr_matrix(
            (data, indices, indptr),
            shape=(n_cells, n_cells),
        )
    except ImportError:
        pass

    idx = knn_indices.astype(np.int64, copy=True)
    dst = knn_distances.astype(np.float32, copy=True)
    pad = knn_indices == MAX
    if pad.any():
        rows = np.broadcast_to(
            np.arange(n_cells, dtype=np.int64)[:, None], idx.shape
        )
        idx[pad] = rows[pad]
        dst[pad] = 0.0

    k_total = idx.shape[1]
    try:
        from umap.umap_ import fuzzy_simplicial_set  # type: ignore

        rng = np.random.default_rng(0)
        conn, _, _ = fuzzy_simplicial_set(
            X=np.zeros((n_cells, 1), dtype=np.float32),
            n_neighbors=k_total,
            random_state=rng,
            metric="euclidean",
            knn_indices=idx,
            knn_dists=dst,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
        )
        return sp.csr_matrix(conn)
    except ImportError:
        pass

    sigmas = np.ones(n_cells, dtype=np.float64)
    for i in range(n_cells):
        row = dst[i]
        nonzero = row[row > 0]
        if len(nonzero) > 0:
            sigmas[i] = float(np.median(nonzero))
    weights = np.exp(-((dst / sigmas[:, None]) ** 2))
    rows = np.broadcast_to(
        np.arange(n_cells, dtype=np.int64)[:, None], idx.shape
    ).ravel()
    csr = sp.csr_matrix(
        (weights.ravel().astype(np.float64), (rows, idx.ravel())),
        shape=(n_cells, n_cells),
    )
    csr.setdiag(0.0)
    csr.eliminate_zeros()
    sym = csr + csr.T
    prod = csr.multiply(csr.T)
    return (sym - prod).tocsr()


def knn_and_fuzzy(
    embedding: np.ndarray,
    batch_codes: np.ndarray,
    neighbors_within_batch: int,
    backend: str,
    metric: str = "cosine",
) -> tuple[dict, Any]:
    """BBKNN-style kNN + fuzzy simplicial set → ``(knn_dict, CSR)``.

    ``metric="cosine"`` L2-normalizes the embedding before passing to the
    euclidean-only kernel. On unit-norm vectors the top-k euclidean
    neighbors are exactly the top-k cosine neighbors — the scanpy trick.
    Critical for scRNA trajectory data.
    """
    from fast_auto_scrna.integration.bbknn import bbknn_kneighbors

    if metric == "cosine":
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embedding = (embedding / norms).astype(np.float32, copy=False)
    elif metric != "euclidean":
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")

    knn = bbknn_kneighbors(
        embedding, batch_codes,
        neighbors_within_batch=int(neighbors_within_batch),
        backend=backend,
    )
    conn = fuzzy_connectivities(
        knn["indices"], knn["distances"], embedding.shape[0],
    )
    return knn, conn

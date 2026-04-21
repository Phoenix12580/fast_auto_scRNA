"""Integration / batch-correction methods.

`bbknn` mirrors the per-batch kNN construction from the `bbknn` Python
package (Park et al. 2020). The Rust kernel returns exact brute-force
neighbors; connectivities are built with UMAP's fuzzy simplicial set when
`umap-learn` is available, falling back to plain Gaussian-kernel weights
otherwise.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from scatlas._scatlas_native.ext import (
    bbknn_kneighbors as _bbknn_kneighbors,
    harmony_integrate as _harmony_integrate,
)

__all__ = ["bbknn_kneighbors", "bbknn", "harmony"]


def bbknn_kneighbors(
    pca: np.ndarray,
    batch_labels: np.ndarray,
    neighbors_within_batch: int = 3,
    backend: str = "auto",
    ef_search: int | None = None,
    auto_threshold: int = 5000,
) -> dict[str, Any]:
    """Batch-balanced k-NN over a PCA embedding.

    Parameters
    ----------
    pca
        ``(n_cells, n_dims)`` float32 PCA matrix (will be cast if needed).
    batch_labels
        Length-n_cells array; int-like stays int32, str-like is label-
        encoded (see ``batch_code_map`` in the return dict).
    neighbors_within_batch
        ``k`` nearest neighbors per batch (bbknn default 3).
    backend
        ``"brute"`` for exact O(N·max_batch) (fast up to ~5k/batch),
        ``"hnsw"`` for approximate O(N·log N) (scales to 1M+),
        ``"auto"`` (default) switches to HNSW when any batch exceeds
        ``auto_threshold``.
    ef_search
        HNSW query breadth (ignored when backend is "brute").  ``None``
        picks ``max(32, 4*neighbors_within_batch)``, which keeps recall
        >= 0.95 vs brute on 30-D PCA.
    auto_threshold
        Largest-batch-size threshold for switching to HNSW in ``auto``
        mode.  Default 5000 reflects the crossover where HNSW build +
        query beats brute O(N*max_batch).

    Returns
    -------
    dict with keys ``indices``, ``distances``, ``batches``,
    ``batch_code_map``, ``backend_used``.
    """
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

    # Resolve 'auto' on the Python side too so we can report `backend_used`.
    backend_used = backend
    if backend == "auto":
        _, counts = np.unique(batch_int32, return_counts=True)
        backend_used = "hnsw" if counts.max() >= auto_threshold else "brute"

    idx, dist, batches = _bbknn_kneighbors(
        pca_f32,
        batch_int32,
        int(neighbors_within_batch),
        backend,
        ef_search,
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
    indices: np.ndarray, distances: np.ndarray, n_cells: int
) -> sp.csr_matrix:
    """Pack the (n_cells, k_total) kNN output into a sparse distances matrix.

    Drops the ``u32::MAX`` padding slots so small batches produce fewer
    edges. The result is an *asymmetric* distance matrix; symmetrization
    is a property of downstream UMAP connectivity construction, not of
    the raw kNN output.
    """
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
    ``adata.obsp[f'{key}_connectivities']`` (when ``with_connectivities``)
    plus a ``adata.uns[key]`` metadata stub mirroring scanpy's neighbors
    convention, so downstream ``sc.tl.umap`` / ``sc.tl.leiden`` can consume
    the result directly.
    """
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
    conn_csr: sp.csr_matrix | None = None
    if with_connectivities:
        conn_csr = _fuzzy_connectivities(
            result["indices"], result["distances"], n_cells
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
            "backend": result.get("backend_used", "scatlas-core"),
        },
    }

    return {
        "distances": dist_csr,
        "connectivities": conn_csr,
        "batches": result["batches"],
        "batch_code_map": result["batch_code_map"],
    }


def harmony(
    adata,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
    key_added: str = "X_pca_harmony",
    n_clusters: int | None = None,
    theta: float = 2.0,
    sigma: float = 0.1,
    lambda_: float | None = 1.0,
    alpha: float = 0.2,
    max_iter: int = 10,
    max_iter_cluster: int = 20,
    epsilon_cluster: float = 1e-3,
    epsilon_harmony: float = 1e-2,
    block_size: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Run Harmony 2.0 on an embedding stored in ``adata.obsm[use_rep]``.

    Writes the corrected embedding to ``adata.obsm[key_added]`` (default
    ``X_pca_harmony``), records run metadata in ``adata.uns['harmony']``.

    Defaults match R's ``RunHarmony()`` (theta=2, sigma=0.1, lambda=1,
    max_iter=10, block_size=0.05, epsilon_cluster=1e-3,
    epsilon_harmony=1e-2). ``n_clusters`` defaults to
    ``min(round(N/30), 100)``.

    Set ``lambda_=None`` to enable **dynamic-lambda** mode — matches
    `RunHarmony(lambda=NULL)`, per-cluster ridge is
    ``lambda_kb = alpha · E[k, b]`` recomputed each correction step
    (protects against over-correction for small batches).
    """
    if use_rep not in adata.obsm:
        raise KeyError(
            f"{use_rep!r} not in adata.obsm — run PCA first or pass use_rep"
        )
    if batch_key not in adata.obs:
        raise KeyError(f"{batch_key!r} not in adata.obs")

    pca = np.ascontiguousarray(adata.obsm[use_rep], dtype=np.float32)

    batch_raw = adata.obs[batch_key].to_numpy()
    code_map: dict[int, Any] = {}
    if np.asarray(batch_raw).dtype.kind in {"i", "u"}:
        batch_codes = np.asarray(batch_raw, dtype=np.int32)
    else:
        uniq, codes = np.unique(batch_raw, return_inverse=True)
        batch_codes = codes.astype(np.int32)
        code_map = {int(i): uniq[i] for i in range(len(uniq))}

    if len(np.unique(batch_codes)) < 2:
        raise ValueError(
            f"batch_key {batch_key!r} has only {len(np.unique(batch_codes))} "
            "unique value(s); Harmony needs ≥ 2 batches to integrate."
        )

    z_corr, r, y, obj, converged = _harmony_integrate(
        pca,
        batch_codes,
        n_clusters,
        float(theta),
        float(sigma),
        None if lambda_ is None else float(lambda_),
        float(alpha),
        int(max_iter),
        int(max_iter_cluster),
        float(epsilon_cluster),
        float(epsilon_harmony),
        float(block_size),
        int(seed),
    )

    adata.obsm[key_added] = z_corr
    adata.uns["harmony"] = {
        "key_added": key_added,
        "use_rep": use_rep,
        "batch_key": batch_key,
        "batch_code_map": code_map,
        "objective_harmony": np.asarray(obj),
        "converged_at_iter": converged,
        "params": {
            "n_clusters": int(n_clusters) if n_clusters is not None else None,
            "theta": float(theta),
            "sigma": float(sigma),
            "lambda": None if lambda_ is None else float(lambda_),
            "alpha": float(alpha),
            "lambda_mode": "dynamic" if lambda_ is None else "fixed",
            "max_iter": int(max_iter),
            "max_iter_cluster": int(max_iter_cluster),
            "epsilon_cluster": float(epsilon_cluster),
            "epsilon_harmony": float(epsilon_harmony),
            "block_size": float(block_size),
            "seed": int(seed),
        },
    }
    return {
        "z_corrected": z_corr,
        "r": r,
        "y": y,
        "objective_harmony": np.asarray(obj),
        "converged_at_iter": converged,
        "batch_code_map": code_map,
    }


def _fuzzy_connectivities(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_cells: int,
) -> sp.csr_matrix:
    """UMAP's ``fuzzy_simplicial_set`` from our raw kNN arrays.

    Falls back to a per-cell Gaussian kernel when umap-learn isn't
    installed. scatlas pads small batches with ``u32::MAX`` / ``inf``; we
    replace those slots with self-references (distance 0) so every row
    has the same k, which is what umap-learn expects. Self-edges drop
    out of the fuzzy simplicial set automatically.
    """
    MAX = np.iinfo(np.uint32).max
    # Rust fuzzy path: handles u32::MAX sentinels internally, faster
    # than umap-learn on all-threads machines.
    try:
        from scatlas._scatlas_native.ext import fuzzy_simplicial_set as _rust_fuzzy

        idx_u32 = np.ascontiguousarray(knn_indices, dtype=np.uint32)
        dst_f32 = np.ascontiguousarray(knn_distances, dtype=np.float32)
        k_total = idx_u32.shape[1]
        indptr, indices, data = _rust_fuzzy(
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

    # Fallback to umap-learn if Rust binding is missing.
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
        conn, _sigmas, _rhos = fuzzy_simplicial_set(
            X=np.zeros((n_cells, 1), dtype=np.float32),  # ignored when knn_* given
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

    # Gaussian fallback: per-cell bandwidth = median of its non-zero
    # distances; symmetrize by fuzzy union (a ∨ b = a + b - a*b).
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

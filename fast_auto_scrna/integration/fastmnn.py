"""fastMNN batch correction (Haghverdi 2018; batchelor::fastMNN port).

Pure-Python implementation built on ``hnswlib`` for cross-batch kNN +
numpy for correction. Written 2026-04-25 because mnnpy is unmaintained
and won't build on Windows MSVC (Cython _utils uses GCC-only compile
flags).

Algorithm (per Haghverdi 2018):

  1. Cosine-normalize the embedding.
  2. Sort batches by size descending; the largest is the reference.
  3. For each remaining batch b, in order:
     a. Find mutual nearest neighbors (MNN) between b and the running
        reference using k-nearest-neighbors in cosine space (HNSW).
     b. Compute correction vectors at MNN pairs: ``ref_pos - b_pos``.
     c. Smooth correction over all cells in b with a Gaussian kernel
        whose bandwidth = median MNN-pair cosine distance × sigma_scale.
     d. Apply correction: ``b_corrected = b + smoothed_correction``.
     e. Append corrected b to the reference and continue.
  4. Return the corrected embedding (same shape as input).

For 222k cells × 10 batches × 50 PCs the wall is ~5-15 min on a 16-core
CPU, dominated by HNSW queries. Scales roughly linearly with N.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _cosine_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows so euclidean kNN ≡ cosine kNN."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (X / norms).astype(np.float32, copy=False)


def _hnsw_query(
    index_data: np.ndarray, query_data: np.ndarray, k: int,
    ef: int = 128, n_threads: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build HNSW index on ``index_data``, query top-k for ``query_data``.

    Both inputs assumed L2-normalized. Returns (indices, sq_distances)
    where indices are into ``index_data``.
    """
    import hnswlib

    n_idx, dim = index_data.shape
    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=n_idx, ef_construction=200, M=16)
    if n_threads > 0:
        p.set_num_threads(n_threads)
    p.add_items(index_data, np.arange(n_idx, dtype=np.int64))
    p.set_ef(max(ef, k + 1))
    indices, sq_distances = p.knn_query(query_data, k=k)
    return indices.astype(np.int64), sq_distances.astype(np.float32)


def _find_mnn_pairs(
    ref_emb: np.ndarray, b_emb: np.ndarray, k: int,
    n_threads: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find mutual nearest neighbors between two cosine-normalized sets.

    Returns ``(ref_idx, b_idx, sq_dists)`` of length n_pairs — indices
    into ref_emb / b_emb of the cells that appear in each other's top-k.
    """
    nbr_b_in_ref, dist_b_in_ref = _hnsw_query(
        ref_emb, b_emb, k=k, n_threads=n_threads,
    )
    nbr_ref_in_b, _ = _hnsw_query(
        b_emb, ref_emb, k=k, n_threads=n_threads,
    )

    ref_pairs: list[int] = []
    b_pairs: list[int] = []
    dist_pairs: list[float] = []

    ref_neighbor_sets = [set(nbr_ref_in_b[i].tolist()) for i in range(len(ref_emb))]
    for bi in range(len(b_emb)):
        for slot, ri in enumerate(nbr_b_in_ref[bi]):
            ri = int(ri)
            if bi in ref_neighbor_sets[ri]:
                ref_pairs.append(ri)
                b_pairs.append(bi)
                dist_pairs.append(float(dist_b_in_ref[bi, slot]))
    return (
        np.asarray(ref_pairs, dtype=np.int64),
        np.asarray(b_pairs, dtype=np.int64),
        np.asarray(dist_pairs, dtype=np.float32),
    )


def _gaussian_smooth_correction(
    b_emb: np.ndarray,
    b_pair_indices: np.ndarray,
    correction_at_pairs: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Per-cell smoothed correction via Gaussian kernel over MNN pairs.

    Uses cosine distance (dot-product on unit vectors) between each cell
    and the MNN-pair anchor cells. Output shape: ``(n_b, d)``.
    """
    anchor_emb = b_emb[b_pair_indices]                 # (n_pairs, d)
    sims = b_emb @ anchor_emb.T                        # (n_b, n_pairs)
    sims = np.clip(sims, -1.0, 1.0)
    cos_dist = 1.0 - sims                              # in [0, 2]
    # Gaussian on cosine distance.
    weights = np.exp(-(cos_dist ** 2) / (2.0 * sigma * sigma))
    weight_sum = weights.sum(axis=1, keepdims=True) + 1e-12
    weights = weights / weight_sum
    return weights @ correction_at_pairs               # (n_b, d)


def fastmnn(
    pca: np.ndarray,
    batch_codes: np.ndarray,
    *,
    n_neighbors: int = 20,
    sigma_scale: float = 1.0,
    n_threads: int = -1,
) -> dict[str, Any]:
    """Run fastMNN on a PCA embedding. Returns a dict with the corrected
    embedding under ``corrected`` plus diagnostic counts.

    Parameters
    ----------
    pca : (N, d) array
        Input PCA embedding (will be cosine-normalized internally).
    batch_codes : (N,) array
        Batch labels (any hashable / int).
    n_neighbors : int, default 20
        k for MNN search. Standard fastMNN uses 20.
    sigma_scale : float, default 1.0
        Multiplier on the Gaussian kernel bandwidth (=sigma_scale ×
        median MNN-pair cosine distance per batch).
    n_threads : int, default -1
        Threads for HNSW. -1 → use all cores.
    """
    if pca.ndim != 2:
        raise ValueError(f"pca must be 2D, got shape {pca.shape}")
    if len(batch_codes) != pca.shape[0]:
        raise ValueError(
            f"batch_codes length {len(batch_codes)} != pca.shape[0]={pca.shape[0]}"
        )

    norm = _cosine_normalize(pca)

    batches, counts = np.unique(batch_codes, return_counts=True)
    order = batches[np.argsort(-counts)]
    if len(order) == 1:
        return {
            "corrected": pca.astype(np.float32, copy=True),
            "n_pairs_per_merge": [],
            "merge_order": [str(order[0])],
            "skipped_batches": [],
        }

    corrected = norm.copy()
    ref_mask = batch_codes == order[0]
    n_pairs_per_merge: list[int] = []
    skipped: list[Any] = []

    for b_id in order[1:]:
        b_mask = batch_codes == b_id
        ref_emb = corrected[ref_mask]
        b_emb = corrected[b_mask]

        ref_pair_idx, b_pair_idx, sq_dists = _find_mnn_pairs(
            ref_emb, b_emb, k=n_neighbors, n_threads=n_threads,
        )
        if len(ref_pair_idx) == 0:
            skipped.append(b_id)
            n_pairs_per_merge.append(0)
            ref_mask |= b_mask
            continue

        # Cosine distance from squared-L2 on unit vectors:
        #   ||a - b||² = 2 (1 - cos_sim)  ⇒  cos_dist = sq_dist / 2
        cos_dists = sq_dists / 2.0
        med_dist = float(np.median(cos_dists))
        sigma = max(med_dist * sigma_scale, 1e-3)

        correction_at_pairs = ref_emb[ref_pair_idx] - b_emb[b_pair_idx]
        smoothed = _gaussian_smooth_correction(
            b_emb, b_pair_idx, correction_at_pairs, sigma=sigma,
        )
        corrected[b_mask] = b_emb + smoothed
        n_pairs_per_merge.append(int(len(ref_pair_idx)))
        ref_mask |= b_mask

    return {
        "corrected": corrected.astype(np.float32, copy=False),
        "n_pairs_per_merge": n_pairs_per_merge,
        "merge_order": [str(b) for b in order.tolist()],
        "skipped_batches": skipped,
    }

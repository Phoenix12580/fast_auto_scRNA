"""scib-metrics parity interface.

Currently exposed kernels:

* :func:`ilisi` / :func:`clisi` — Local Inverse Simpson's Index over
  batch / cell-type labels, Gaussian-weighted with perplexity calibration.
* :func:`graph_connectivity` — mean per-label largest-CC fraction on a
  k-NN graph.
* :func:`scib_score` — thin aggregate that composes the above plus
  optional sklearn silhouette (scib-metrics' own kBET / silhouette
  routines stay Python-side for now).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from scatlas._scatlas_native.metrics import (
    graph_connectivity_score as _graph_connectivity_score,
    kbet_chi2 as _kbet_chi2,
    lisi_per_cell as _lisi_per_cell,
)

__all__ = [
    "ilisi",
    "clisi",
    "lisi",
    "graph_connectivity",
    "kbet",
    "label_silhouette",
    "batch_silhouette",
    "scib_score",
]


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    """Label-encode arbitrary dtype to int32, using i32::MIN for NaN / missing."""
    arr = np.asarray(labels)
    if arr.dtype.kind in {"i", "u"}:
        return arr.astype(np.int32)
    uniq, codes = np.unique(arr, return_inverse=True)
    # Reserve i32::MIN as "missing" sentinel (LISI kernel ignores it).
    assert len(uniq) < (1 << 31) - 1, "too many distinct labels"
    return codes.astype(np.int32)


def lisi(
    knn_distances: np.ndarray,
    knn_labels: np.ndarray,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Per-cell Local Inverse Simpson's Index.

    Parameters
    ----------
    knn_distances
        ``(n_cells, k)`` float32 — distances to k neighbors (typically
        Euclidean on PCA).
    knn_labels
        ``(n_cells, k)`` — label code of each neighbor. Padding slots
        (``u32::MAX`` distances → any label) are safe; if your pipeline
        fills with sentinels, use ``i32::MIN`` or numpy NaN on the label
        side.
    perplexity
        Gaussian-kernel perplexity. Default 30 matches
        ``scib_metrics.ilisi_knn`` / harmonypy.

    Returns
    -------
    ``(n_cells,)`` float32. Range ``[1, #distinct_labels]``.
    """
    d = np.ascontiguousarray(knn_distances, dtype=np.float32)
    if knn_labels.dtype == np.int32:
        l = np.ascontiguousarray(knn_labels)
    else:
        flat = _encode_labels(np.asarray(knn_labels).ravel())
        l = flat.reshape(knn_labels.shape)
    return _lisi_per_cell(d, l, float(perplexity))


def ilisi(
    knn_distances: np.ndarray,
    knn_batch_labels: np.ndarray,
    perplexity: float = 30.0,
) -> float:
    """Dataset-level iLISI: mean over cells, rescaled to ``[0, 1]``.

    Normalized so that perfect batch mixing across ``n_batches`` maps to
    1 and no mixing maps to 0 (same convention as scib_metrics).
    """
    per_cell = lisi(knn_distances, knn_batch_labels, perplexity)
    n_batches = len(np.unique(knn_batch_labels))
    if n_batches <= 1:
        return 1.0
    # scib-metrics: (mean(LISI) - 1) / (n_batches - 1), clamped to [0, 1]
    raw = (per_cell.mean() - 1.0) / (n_batches - 1)
    return float(np.clip(raw, 0.0, 1.0))


def clisi(
    knn_distances: np.ndarray,
    knn_label_labels: np.ndarray,
    perplexity: float = 30.0,
) -> float:
    """Dataset-level cLISI: mean LISI over cell-type labels, rescaled so
    that *lower* LISI (more cell-type-coherent neighborhoods) → higher
    score. Output in ``[0, 1]``; 1 = perfect preservation.
    """
    per_cell = lisi(knn_distances, knn_label_labels, perplexity)
    n_lab = len(np.unique(knn_label_labels))
    if n_lab <= 1:
        return 1.0
    # scib-metrics: 1 - (mean(LISI) - 1) / (n_labels - 1), clamped to [0, 1]
    raw = 1.0 - (per_cell.mean() - 1.0) / (n_lab - 1)
    return float(np.clip(raw, 0.0, 1.0))


def graph_connectivity(
    knn_indices: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean per-label largest-CC fraction on the supplied k-NN graph.

    Also accepts a sparse ``scipy.sparse`` matrix in place of the dense
    ``(n_cells, k)`` indices array; in that case the non-zero column
    indices per row are used as neighbors (matches scanpy's
    ``adata.obsp['connectivities']`` layout).
    """
    if sp.issparse(knn_indices):
        csr = knn_indices.tocsr()
        n_cells = csr.shape[0]
        # Convert to dense (n_cells, max_k) with u32::MAX padding
        max_k = int(csr.getnnz(axis=1).max()) if csr.nnz > 0 else 0
        if max_k == 0:
            return 0.0
        idx = np.full((n_cells, max_k), np.iinfo(np.uint32).max, dtype=np.uint32)
        for i in range(n_cells):
            start, end = csr.indptr[i], csr.indptr[i + 1]
            cols = csr.indices[start:end]
            idx[i, : len(cols)] = cols.astype(np.uint32)
    else:
        idx = np.ascontiguousarray(knn_indices, dtype=np.uint32)
    lbl = _encode_labels(labels)
    return _graph_connectivity_score(idx, lbl)


def kbet(
    knn_indices: np.ndarray,
    batch_labels: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """k-nearest-neighbor batch effect test (Büttner 2019).

    For each cell, tests whether its neighbor batch distribution matches
    the global batch distribution via χ² goodness-of-fit. Returns
    per-cell χ² / p-values plus the dataset-level **acceptance rate**
    (fraction of cells whose batch composition is indistinguishable from
    the global mix at level ``alpha``).

    Parameters
    ----------
    knn_indices
        ``(n_cells, k)`` uint32 — typically ``scatlas.ext.bbknn_kneighbors``
        output. ``u32::MAX`` slots are skipped.
    batch_labels
        Length-n_cells array of batch labels (int or string).
    alpha
        Significance threshold for the per-cell test; default 0.05.

    Returns
    -------
    dict with keys ``acceptance_rate`` (higher = better integration;
    1.0 ≈ perfect mix), ``chi2``, ``pvals``, ``n_batches``.
    """
    from scipy.stats import chi2 as _chi2_dist  # lazy — scipy in runtime deps

    encoded = _encode_labels(batch_labels)
    uniq, global_counts = np.unique(encoded, return_counts=True)
    assert np.array_equal(uniq, np.arange(len(uniq))), (
        "encoded labels must be contiguous 0..n_batches-1 after _encode_labels"
    )
    n_batches = len(uniq)
    if n_batches < 2:
        n = knn_indices.shape[0]
        return {
            "acceptance_rate": 1.0,
            "chi2": np.zeros(n),
            "pvals": np.ones(n),
            "n_batches": n_batches,
        }

    # Gather per-neighbor batch codes, replace padding with i32::MIN sentinel
    MAX = np.iinfo(np.uint32).max
    idx = knn_indices.astype(np.int64, copy=True)
    pad = knn_indices == MAX
    if pad.any():
        # Point padded slots at cell 0 — the i32::MIN override makes it safe
        idx[pad] = 0
    nbr_batch = encoded[idx].astype(np.int32, copy=True)
    if pad.any():
        nbr_batch[pad] = np.iinfo(np.int32).min

    chi2 = _kbet_chi2(
        np.ascontiguousarray(nbr_batch),
        np.ascontiguousarray(global_counts.astype(np.uint64)),
    )
    df = n_batches - 1
    pvals = _chi2_dist.sf(chi2, df)
    acceptance = float((pvals >= alpha).mean())
    return {
        "acceptance_rate": acceptance,
        "chi2": chi2,
        "pvals": pvals,
        "n_batches": n_batches,
    }


def label_silhouette(
    embedding: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Label ASW rescaled to [0, 1]. Higher = cell-type structure preserved.

    Thin wrapper over ``sklearn.metrics.silhouette_score``; returns
    ``(asw + 1) / 2`` to match scib_metrics' ``silhouette_label``.
    """
    from sklearn.metrics import silhouette_score  # type: ignore

    if len(np.unique(labels)) < 2:
        return 1.0
    s = silhouette_score(embedding, labels, metric=metric)
    return float((s + 1.0) / 2.0)


def batch_silhouette(
    embedding: np.ndarray,
    batch_labels: np.ndarray,
    cell_type_labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Per-cell-type batch ASW, rescaled so higher = batches are mixed.

    scib formulation: within each cell-type group compute silhouette on
    batch labels, then ``1 - |mean(s)|``; average across cell types.
    Score 1.0 means batches are indistinguishable within each cell type.

    Cell-type groups with fewer than 2 batches (or 2 cells per batch) are
    skipped — matches scib-metrics behavior.
    """
    from sklearn.metrics import silhouette_samples  # type: ignore

    ct = np.asarray(cell_type_labels)
    bt = np.asarray(batch_labels)
    scores: list[float] = []
    for c in np.unique(ct):
        mask = ct == c
        bsub = bt[mask]
        uniq, cnt = np.unique(bsub, return_counts=True)
        if len(uniq) < 2 or int(cnt.min()) < 2:
            continue
        s = silhouette_samples(embedding[mask], bsub, metric=metric)
        scores.append(1.0 - float(np.abs(np.mean(s))))
    if not scores:
        return 1.0
    return float(np.mean(scores))


def scib_score(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    batch_labels: np.ndarray,
    label_labels: np.ndarray,
    perplexity: float = 30.0,
    embedding: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compose iLISI + cLISI + graph_connectivity into a single report.

    Parameters are the raw kNN outputs from ``scatlas.ext.bbknn_kneighbors``
    (or any compatible neighbor source) plus per-cell batch and label
    codes.

    Returns
    -------
    dict with keys ``ilisi``, ``clisi``, ``graph_connectivity``, and a
    simple unweighted ``mean`` of the three — useful as a scalar to
    compare integration methods head-to-head.
    """
    # Promote batch and label arrays to neighbor-shaped via gather.
    n_cells, k_total = knn_indices.shape
    MAX = np.iinfo(np.uint32).max

    batch_codes = _encode_labels(batch_labels)
    label_codes = _encode_labels(label_labels)

    # Build neighbor-label matrices; replace padding slots with i32::MIN.
    idx_safe = knn_indices.copy()
    # Where padding, point at self to avoid out-of-bounds; we'll overwrite
    # with i32::MIN.
    rows = np.broadcast_to(
        np.arange(n_cells, dtype=np.uint32)[:, None], idx_safe.shape
    )
    pad = idx_safe == MAX
    idx_safe = np.where(pad, rows, idx_safe)
    nbr_batch = batch_codes[idx_safe.astype(np.int64)]
    nbr_label = label_codes[idx_safe.astype(np.int64)]
    nbr_batch[pad] = np.iinfo(np.int32).min
    nbr_label[pad] = np.iinfo(np.int32).min

    ilisi_val = ilisi(knn_distances, nbr_batch, perplexity)
    clisi_val = clisi(knn_distances, nbr_label, perplexity)
    gc_val = graph_connectivity(knn_indices, label_labels)
    kbet_val = kbet(knn_indices, batch_labels)["acceptance_rate"]

    result: dict[str, Any] = {
        "ilisi": ilisi_val,
        "clisi": clisi_val,
        "graph_connectivity": gc_val,
        "kbet_acceptance": kbet_val,
    }
    if embedding is not None:
        result["label_silhouette"] = label_silhouette(embedding, label_labels)
        result["batch_silhouette"] = batch_silhouette(
            embedding, batch_labels, label_labels
        )
    result["mean"] = float(np.mean(list(result.values())))
    return result

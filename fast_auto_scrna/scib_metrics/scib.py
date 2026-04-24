"""scIB-style metrics composed from Rust kernels. Ported from v1
``scatlas.metrics`` at V2-P2 (the SCCAF path lives in ``sccaf.py``)."""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    """Label-encode arbitrary dtype to int32."""
    arr = np.asarray(labels)
    if arr.dtype.kind in {"i", "u"}:
        return arr.astype(np.int32)
    uniq, codes = np.unique(arr, return_inverse=True)
    assert len(uniq) < (1 << 31) - 1, "too many distinct labels"
    return codes.astype(np.int32)


def lisi(
    knn_distances: np.ndarray,
    knn_labels: np.ndarray,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Per-cell Local Inverse Simpson's Index."""
    from fast_auto_scrna._native import metrics as _native_metrics

    d = np.ascontiguousarray(knn_distances, dtype=np.float32)
    if knn_labels.dtype == np.int32:
        l = np.ascontiguousarray(knn_labels)
    else:
        flat = _encode_labels(np.asarray(knn_labels).ravel())
        l = flat.reshape(knn_labels.shape)
    return _native_metrics.lisi_per_cell(d, l, float(perplexity))


def ilisi(
    knn_distances: np.ndarray,
    knn_batch_labels: np.ndarray,
    perplexity: float = 30.0,
) -> float:
    """Dataset-level iLISI: mean over cells, rescaled to ``[0, 1]``."""
    per_cell = lisi(knn_distances, knn_batch_labels, perplexity)
    n_batches = len(np.unique(knn_batch_labels))
    if n_batches <= 1:
        return 1.0
    raw = (per_cell.mean() - 1.0) / (n_batches - 1)
    return float(np.clip(raw, 0.0, 1.0))


def clisi(
    knn_distances: np.ndarray,
    knn_label_labels: np.ndarray,
    perplexity: float = 30.0,
) -> float:
    """Dataset-level cLISI: lower LISI → higher score (more cell-type coherent)."""
    per_cell = lisi(knn_distances, knn_label_labels, perplexity)
    n_lab = len(np.unique(knn_label_labels))
    if n_lab <= 1:
        return 1.0
    raw = 1.0 - (per_cell.mean() - 1.0) / (n_lab - 1)
    return float(np.clip(raw, 0.0, 1.0))


def graph_connectivity(
    knn_indices: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean per-label largest-CC fraction on the supplied k-NN graph."""
    from fast_auto_scrna._native import metrics as _native_metrics

    if sp.issparse(knn_indices):
        csr = knn_indices.tocsr()
        n_cells = csr.shape[0]
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
    return _native_metrics.graph_connectivity_score(idx, lbl)


def kbet(
    knn_indices: np.ndarray,
    batch_labels: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """k-nearest-neighbor batch effect test (Büttner 2019)."""
    from fast_auto_scrna._native import metrics as _native_metrics
    from scipy.stats import chi2 as _chi2_dist

    encoded = _encode_labels(batch_labels)
    uniq, global_counts = np.unique(encoded, return_counts=True)
    assert np.array_equal(uniq, np.arange(len(uniq))), (
        "encoded labels must be contiguous 0..n_batches-1"
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

    MAX = np.iinfo(np.uint32).max
    idx = knn_indices.astype(np.int64, copy=True)
    pad = knn_indices == MAX
    if pad.any():
        idx[pad] = 0
    nbr_batch = encoded[idx].astype(np.int32, copy=True)
    if pad.any():
        nbr_batch[pad] = np.iinfo(np.int32).min

    chi2 = _native_metrics.kbet_chi2(
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
    """Label ASW rescaled to ``[0, 1]``. Higher = cell-type structure preserved."""
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2:
        return 1.0
    s = silhouette_score(embedding, labels, metric=metric)
    return float((s + 1.0) / 2.0)


def isolated_label_silhouette(
    embedding: np.ndarray,
    labels: np.ndarray,
    batch_labels: np.ndarray,
    *,
    iso_threshold: int | None = None,
    metric: str = "euclidean",
) -> float:
    """Isolated-label ASW — scib-metrics' ``isolated_labels_asw``."""
    from sklearn.metrics import silhouette_score

    lbl = np.asarray(labels)
    bt = np.asarray(batch_labels)
    unique_labels = np.unique(lbl)
    batches_per_label = np.array([
        len(np.unique(bt[lbl == L])) for L in unique_labels
    ])
    if iso_threshold is None:
        iso_threshold = int(batches_per_label.min())
    iso = unique_labels[batches_per_label <= iso_threshold]
    if len(iso) == 0:
        return 1.0

    scores: list[float] = []
    for L in iso:
        mask = (lbl == L).astype(np.int32)
        if mask.sum() < 2 or mask.sum() == len(lbl):
            continue
        s = silhouette_score(embedding, mask, metric=metric)
        scores.append((s + 1.0) / 2.0)
    if not scores:
        return 1.0
    return float(np.mean(scores))


def batch_silhouette(
    embedding: np.ndarray,
    batch_labels: np.ndarray,
    cell_type_labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Per-cell-type batch ASW, rescaled so higher = batches are mixed."""
    from sklearn.metrics import silhouette_samples

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
    """Compose iLISI + cLISI + graph_connectivity + kBET (+ silhouettes if
    ``embedding`` is given) into a single report."""
    n_cells, k_total = knn_indices.shape
    MAX = np.iinfo(np.uint32).max

    batch_codes = _encode_labels(batch_labels)
    label_codes = _encode_labels(label_labels)

    idx_safe = knn_indices.copy()
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
        result["isolated_label"] = isolated_label_silhouette(
            embedding, label_labels, batch_labels,
        )
    result["mean"] = float(np.mean(list(result.values())))
    return result

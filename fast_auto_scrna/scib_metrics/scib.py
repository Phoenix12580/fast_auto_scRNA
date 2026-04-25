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
    """k-nearest-neighbor batch effect test (Büttner 2019).

    Detects batch-balanced kNN outputs (BBKNN-style: every cell has
    identical per-batch neighbor counts regardless of global batch
    abundance). On such graphs kBET's null hypothesis ("neighborhood
    batch distribution matches global") is violated by construction, so
    we return NaN with a ``note`` field rather than a meaningless 0.
    For Harmony / none routes (plain kNN) the test runs normally.
    """
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

    # ── Batch-balanced detection ───────────────────────────────────────
    # Sample up to 100 cells' per-batch neighbor counts. If they're
    # identical across cells the graph is batch-balanced (BBKNN) and
    # kBET is not meaningful. Cheap O(100 * k) sample, bails early.
    n_cells = nbr_batch.shape[0]
    sample_n = min(100, n_cells)
    if sample_n >= 2:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_cells, sample_n, replace=False)
        sample_counts = np.zeros((sample_n, n_batches), dtype=np.int32)
        for i_out, i in enumerate(sample_idx):
            row = nbr_batch[i]
            valid = row[row != np.iinfo(np.int32).min]
            if len(valid) > 0:
                bc = np.bincount(valid, minlength=n_batches)
                sample_counts[i_out] = bc[:n_batches]
        col_std = sample_counts.std(axis=0).max() if sample_counts.size else 0.0
        if col_std < 0.01:
            n = knn_indices.shape[0]
            return {
                "acceptance_rate": float("nan"),
                "chi2": np.full(n, float("nan")),
                "pvals": np.full(n, float("nan")),
                "n_batches": n_batches,
                "note": (
                    "batch-balanced kNN detected (constant neighbor composition "
                    "across cells); kBET null hypothesis violated by construction "
                    "— use iLISI for batch-mixing on BBKNN-style graphs"
                ),
            }

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
    embedding: np.ndarray, labels: np.ndarray,
) -> float:
    """Label ASW rescaled to [0, 1]. Higher = cell-type structure preserved.

    Backed by ``scib_metrics.silhouette_label`` (JAX-jitted, chunked pairwise)
    — atlas-scale-capable. ~5 min for 222k cells × 20 dims on a 16-core CPU.
    Identical numeric output to ``sklearn.metrics.silhouette_score`` (then
    rescaled to ``(s + 1) / 2``).
    """
    import scib_metrics as _sm
    if len(np.unique(labels)) < 2:
        return 1.0
    X = np.ascontiguousarray(embedding, dtype=np.float32)
    return float(_sm.silhouette_label(X, np.asarray(labels), rescale=True))


def batch_silhouette(
    embedding: np.ndarray,
    batch_labels: np.ndarray,
    cell_type_labels: np.ndarray,
) -> float:
    """Per-cell-type batch ASW, rescaled so higher = batches mixed within type.

    Backed by ``scib_metrics.silhouette_batch``. Same atlas-scale story as
    ``label_silhouette``.
    """
    import scib_metrics as _sm
    X = np.ascontiguousarray(embedding, dtype=np.float32)
    return float(_sm.silhouette_batch(
        X, np.asarray(cell_type_labels), np.asarray(batch_labels),
        rescale=True,
    ))


def isolated_label_silhouette(
    embedding: np.ndarray,
    labels: np.ndarray,
    batch_labels: np.ndarray,
    iso_threshold: int | None = None,
) -> float:
    """Isolated-label ASW (scib-metrics' ``isolated_labels``).

    Score how well rare labels (those appearing in ``<= iso_threshold``
    batches) are isolated in the embedding. ``iso_threshold=None`` → use
    the minimum across labels (scib default).
    """
    import scib_metrics as _sm
    X = np.ascontiguousarray(embedding, dtype=np.float32)
    return float(_sm.isolated_labels(
        X, np.asarray(labels), np.asarray(batch_labels),
        rescale=True, iso_threshold=iso_threshold,
    ))


def scib_score(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    batch_labels: np.ndarray,
    label_labels: np.ndarray,
    perplexity: float = 30.0,
    embedding: np.ndarray | None = None,
    compute_kbet: bool = False,
) -> dict[str, Any]:
    """Compose iLISI + cLISI + graph_connectivity + kBET (+ silhouettes
    when ``embedding`` is given) into a single report.

    The three silhouettes (label / batch / isolated) match the metric
    panel used by Gao et al. Cancer Cell 2024 (cross-tissue fibroblast
    atlas). They were removed in V2-P5 because the sklearn implementation
    was O(N²) on atlas-scale data; re-introduced here via
    ``scib-metrics`` v0.5.1's JAX-jitted chunked implementation, which
    runs in ~5 min per route on 222k cells (vs ~45 min for sklearn).

    Pass ``embedding=None`` to skip the silhouettes (faster, but
    deviates from the paper's panel).
    """
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

    result: dict[str, Any] = {
        "ilisi": ilisi_val,
        "clisi": clisi_val,
        "graph_connectivity": gc_val,
    }
    if compute_kbet:
        kbet_res = kbet(knn_indices, batch_labels)
        result["kbet_acceptance"] = kbet_res["acceptance_rate"]
        if "note" in kbet_res:
            result["kbet_note"] = kbet_res["note"]
    if embedding is not None:
        result["label_silhouette"] = label_silhouette(embedding, label_labels)
        result["batch_silhouette"] = batch_silhouette(
            embedding, batch_labels, label_labels,
        )
        result["isolated_label"] = isolated_label_silhouette(
            embedding, label_labels, batch_labels,
        )
    # nanmean over numeric entries: skips NaN kBET when batch-balanced kNN
    # triggered the bail-out in kbet(). String notes ("kbet_note") are excluded.
    numeric_vals = [v for v in result.values() if isinstance(v, (int, float))]
    result["mean"] = float(np.nanmean(numeric_vals)) if numeric_vals else float("nan")
    return result

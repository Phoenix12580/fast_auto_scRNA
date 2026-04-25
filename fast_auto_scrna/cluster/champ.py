"""CHAMP — Convex Hull of Admissible Modularity Partitions.

Implementation of Weir, Emmons, Wakefield, Hopkins & Mucha (2017),
"Post-processing partitions to identify domains of modularity
optimization" (*Algorithms* 10(3):93), as a drop-in resolution picker
for the fast_auto_scrna pipeline.

Algorithm
---------
For any **fixed** partition P, Newman modularity Q is **linear** in
the resolution parameter γ::

    Q(γ; P) = a_P − γ · b_P

so each candidate partition is a line in the (γ, Q) plane. The
partition that's modularity-optimal at any given γ lies on the
**upper envelope** of those lines, which is dual to the **upper convex
hull** of the points (b_P, a_P).

CHAMP runs Leiden at N candidate γ values, keeps unique partitions,
finds the upper hull of (b, a), computes per-hull-vertex *admissible
γ-range*, and returns the hull partition with the **widest** range.
The result is a partition that is modularity-optimal across the
broadest band of resolutions — i.e., the most "stable" partition by a
purely deterministic, geometric criterion.

Why CHAMP over our knee picker
------------------------------
On 222k v2-P10 baseline, the knee picker spent 28.6 min running 150
Leidens to draw a dense conductance curve, then ran a sampling-
sensitive ``first_plateau`` detector on it. Two attempts (two-stage
coarse-then-fine, range truncation 0..0.40) both produced large
picked_r drift (Δr ≥ 0.15, Δk ≥ 4) because the detector's heuristics
break under sparser sampling.

CHAMP doesn't need a dense curve. It uses 30 Leidens (the ones at
distinct γ that produce distinct partitions), then post-processes
purely linearly via convex hull. Wall is dominated by the Leidens —
on 222k roughly 60-120 s wall in the parallel sweep — and the
selection is a closed-form geometric statement, not a tuned heuristic.

Reference
---------
Weir, Emmons, Wakefield, Hopkins, Mucha. *Algorithms* 10(3):93, 2017.
https://doi.org/10.3390/a10030093
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ────────────────────── modularity coefficients ───────────────────────────────


def _modularity_coefficients(
    W,
    labels: np.ndarray,
    modularity: Literal["newman", "cpm"] = "newman",
) -> tuple[float, float]:
    """Return ``(a, b)`` such that ``Q(γ; P) = a − γ·b``.

    - ``newman`` (Newman-Girvan):
        a = (within-cluster edge weight) / 2m,
        b = Σ_c (Σ_{i∈c} d_i)² / (2m)²
    - ``cpm``    (Constant Potts Model — resolution-limit-free):
        a = (within-cluster edge weight) / W_total,
        b = Σ_c |c|² / N²

    O(nnz(W)) — single COO sweep + scatter-add.
    """
    if not sp.issparse(W):
        W = sp.csr_matrix(W)
    coo = W.tocoo()
    total = float(W.sum())  # 2m for undirected
    if total <= 0:
        return 0.0, 0.0

    same = labels[coo.row] == labels[coo.col]
    a = float(coo.data[same].sum()) / total

    _, inv = np.unique(labels, return_inverse=True)
    if modularity == "newman":
        d = np.asarray(W.sum(axis=1)).ravel()
        D = np.bincount(inv, weights=d)
        b = float(np.sum(D * D)) / (total * total)
    elif modularity == "cpm":
        n = float(W.shape[0])
        sizes = np.bincount(inv).astype(np.float64)
        b = float(np.sum(sizes * sizes)) / (n * n)
    else:
        raise ValueError(
            f"modularity must be 'newman' or 'cpm'; got {modularity!r}."
        )
    return a, b


# ────────────────────── upper convex hull ──────────────────────────────────────


def _upper_hull_indices(b: np.ndarray, a: np.ndarray) -> list[int]:
    """Andrew's monotone-chain upper hull on points (b_i, a_i).

    Returns indices in ascending b. Geometric dual of the upper envelope
    of the lines y = a_i − b_i x.
    """
    n = len(b)
    if n <= 2:
        return list(range(n))
    # Sort by b asc; ties broken by a desc so the highest point at each
    # b stays.
    order = sorted(range(n), key=lambda i: (b[i], -a[i]))
    hull: list[int] = []
    for i in order:
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            cross = (
                (b[i2] - b[i1]) * (a[i] - a[i1])
                - (a[i2] - a[i1]) * (b[i] - b[i1])
            )
            if cross >= 0:
                hull.pop()  # collinear or left turn → not upper hull
            else:
                break
        hull.append(i)
    return hull


# ───────────────────────────── public entry ───────────────────────────────────


def optimize_resolution_champ(
    adata,
    *,
    method: str,
    conn=None,
    resolutions: Sequence[float] | None = None,
    n_partitions: int = 30,
    gamma_min: float = 0.05,
    gamma_max: float = 1.50,
    modularity: Literal["newman", "cpm"] = "newman",
    width_metric: Literal["log", "linear", "relative"] = "log",
    seed: int = 0,
    leiden_flavor: Literal["igraph", "leidenalg"] = "igraph",
    leiden_n_iterations: int = 2,
    max_workers: int | None = None,
    worker_priority: str | None = "below_normal",
    verbose: bool = True,
) -> pd.DataFrame:
    """CHAMP resolution picker — Weir et al. 2017.

    Parameters
    ----------
    adata
        AnnData with the route's neighbor graph already built.
    method
        Route name (e.g. ``"bbknn"``). Used only to look up
        ``adata.obsp[f"{method}_connectivities"]`` if ``conn`` is None.
    conn
        Optional explicit CSR connectivity matrix. If None, falls back
        to ``adata.obsp[f"{method}_connectivities"]``.
    resolutions
        Explicit γ grid; overrides ``n_partitions/gamma_min/gamma_max``.
    n_partitions
        Number of γ values evenly spaced over [gamma_min, gamma_max]
        when ``resolutions`` is None. Default 30 — Weir 2017 uses 30-50;
        going higher mostly produces duplicate partitions.
    gamma_min, gamma_max
        γ scan endpoints. ``gamma_max`` also caps the leftmost hull
        partition's "open" admissible range so widths are finite.
        Default [0.05, 1.50] — atlas-typical knee at r≈0.2-0.4 sits
        comfortably inside.
    modularity
        ``newman`` (default) or ``cpm`` (resolution-limit-free).
    width_metric
        ``log`` (default), ``linear`` (Weir 2017 canonical), or
        ``relative``. log is scale-free under γ→cγ and avoids
        over-rewarding fine partitions.

    Returns
    -------
    DataFrame, one row per UNIQUE candidate partition, columns:
      origin_resolution, a, b, n_clusters, on_hull,
      gamma_lo, gamma_hi, gamma_range, is_picked.
    The chosen partition's labels are stored in
    ``adata.uns[f"_champ_{method}_chosen_labels"]`` (caller pulls them
    out via the ``is_picked`` row).
    """
    from .resolution import _leiden_sweep  # share the parallel sweep infra

    if conn is None:
        graph_key = f"{method}_connectivities"
        if graph_key not in adata.obsp:
            raise KeyError(
                f"Missing {graph_key!r} in adata.obsp. Build the neighbor "
                f"graph first, or pass conn= directly."
            )
        G = adata.obsp[graph_key].tocsr()
    else:
        G = sp.csr_matrix(conn)

    if resolutions is None:
        resolutions = list(np.linspace(gamma_min, gamma_max, n_partitions))
    resolutions = sorted({float(np.round(r, 4)) for r in resolutions})
    if len(resolutions) < 2:
        raise ValueError("CHAMP needs at least 2 distinct γ values.")

    if verbose:
        print(f"    [champ] sweeping {len(resolutions)} γ values "
              f"∈ [{resolutions[0]:.3f}, {resolutions[-1]:.3f}] "
              f"({modularity} modularity, {width_metric} width)")

    # 1. Run Leiden in parallel at every candidate γ.
    labels_by_r = _leiden_sweep(
        G, list(resolutions),
        seed=seed, n_iterations=leiden_n_iterations,
        leiden_flavor=leiden_flavor,
        max_workers=max_workers, worker_priority=worker_priority,
    )

    # 2. Deduplicate partitions on label fingerprint, keep origin γ.
    partitions: list[dict] = []
    seen: dict[bytes, int] = {}
    for r in resolutions:
        lbl = np.ascontiguousarray(labels_by_r[r], dtype=np.int32)
        sig = lbl.tobytes()
        if sig in seen:
            continue
        seen[sig] = len(partitions)
        partitions.append({
            "origin_resolution": r,
            "labels":            lbl,
            "n_clusters":        int(np.unique(lbl).size),
        })
    if verbose:
        print(f"    [champ] {len(partitions)} unique partitions out of "
              f"{len(resolutions)} γ values")

    # 3. Compute (a, b) per unique partition.
    for p in partitions:
        a, b = _modularity_coefficients(G, p["labels"], modularity=modularity)
        p["a"], p["b"] = a, b
    a_vals = np.asarray([p["a"] for p in partitions], dtype=np.float64)
    b_vals = np.asarray([p["b"] for p in partitions], dtype=np.float64)

    # 4. Upper convex hull in (b, a) plane.
    hull_idx = _upper_hull_indices(b_vals, a_vals)
    H = len(hull_idx)
    if verbose:
        print(f"    [champ] {H} partitions on upper convex hull")

    # 5. Crossover γ between consecutive hull vertices: where their lines
    #    intersect → boundary of admissible region.
    crossovers = []
    for k in range(H - 1):
        i, j = hull_idx[k], hull_idx[k + 1]
        denom = b_vals[j] - b_vals[i]
        crossovers.append(
            float((a_vals[j] - a_vals[i]) / denom) if denom != 0 else float("inf")
        )

    # 6. Per-hull-vertex admissible γ-range. Indexing follows ascending b
    #    (= ascending hull_idx position). Leftmost (smallest b → fewest
    #    clusters) is admissible only at γ > first crossover.
    on_hull = np.zeros(len(partitions), dtype=bool)
    on_hull[hull_idx] = True
    gamma_lo = np.full(len(partitions), np.nan)
    gamma_hi = np.full(len(partitions), np.nan)
    for k, hull_pos in enumerate(hull_idx):
        if H == 1:
            lo, hi = 0.0, gamma_max
        elif k == 0:
            lo, hi = crossovers[0], gamma_max
        elif k == H - 1:
            lo, hi = 0.0, crossovers[-1]
        else:
            lo, hi = crossovers[k], crossovers[k - 1]
        gamma_lo[hull_pos] = min(gamma_max, max(0.0, lo))
        gamma_hi[hull_pos] = max(0.0, min(gamma_max, hi))

    # 7. Width metric → pick widest admissible range.
    if width_metric == "linear":
        raw_widths = gamma_hi - gamma_lo
    elif width_metric == "log":
        # γ space is scale-free; clamp γ_lo to gamma_min so log is finite.
        lo_c = np.maximum(gamma_lo, gamma_min)
        hi_c = np.maximum(gamma_hi, lo_c)
        raw_widths = np.log(hi_c) - np.log(lo_c)
    elif width_metric == "relative":
        mid = np.maximum((gamma_hi + gamma_lo) / 2.0, gamma_min)
        raw_widths = (gamma_hi - gamma_lo) / mid
    else:
        raise ValueError(
            f"width_metric must be 'log' (default), 'linear', or "
            f"'relative'; got {width_metric!r}."
        )
    widths = np.where(
        on_hull,
        np.maximum(raw_widths, 0.0),
        np.full(len(partitions), -np.inf),
    )
    best_idx = int(np.argmax(widths))

    is_picked = np.zeros(len(partitions), dtype=bool)
    is_picked[best_idx] = True

    # Cache the chosen partition's labels for the caller — avoids
    # re-running leiden at the picked γ.
    adata.uns[f"_champ_{method}_chosen_labels"] = (
        partitions[best_idx]["labels"].copy()
    )

    df = pd.DataFrame({
        "origin_resolution": [p["origin_resolution"] for p in partitions],
        "a":                 a_vals,
        "b":                 b_vals,
        "n_clusters":        [p["n_clusters"] for p in partitions],
        "on_hull":           on_hull,
        "gamma_lo":          gamma_lo,
        "gamma_hi":          gamma_hi,
        "gamma_range":       widths,
        "is_picked":         is_picked,
    }).sort_values("b").reset_index(drop=True)

    if verbose:
        bp = partitions[best_idx]
        print(f"    [champ] picked γ={bp['origin_resolution']:.3f} "
              f"(k={bp['n_clusters']}, admissible "
              f"γ∈[{gamma_lo[best_idx]:.3f}, {gamma_hi[best_idx]:.3f}], "
              f"{width_metric}-width={raw_widths[best_idx]:.3f})")
    return df

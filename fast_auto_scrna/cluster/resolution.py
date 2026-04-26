"""Resolution selection for Leiden clustering.

v2-P12: collapsed to CHAMP-only (Weir et al. 2017). The previous
``knee`` / ``conductance`` / ``graph_silhouette`` / ``target_n``
optimizers were removed after evidence (commit 711223f bench logs +
memory/feedback_two_stage_knee_picker.md) showed all of them either:
  * relied on heuristic detectors fragile to sampling/range, or
  * required dense (~150-pt) sweeps with no statistical principle.

CHAMP is deterministic, modularity-principled (Weir 2017
*Algorithms* 10(3):93), runs ~30 Leidens vs the knee picker's 150,
and on the 222k v2-P10 baseline picks a partition (k=8) that aligns
better with the GT ct.sub structure (k=7) than the old knee pick
(k=12). Implementation lives in ``cluster/champ.py``.

This module keeps the parallel Leiden sweep infrastructure
(``_leiden_sweep``) used by CHAMP, plus ``mean_conductance`` (still
useful as a diagnostic), and the public ``auto_resolution`` entry
that the runner calls.

GS-4 (2026-04-24): parallel Leiden sweep via ProcessPoolExecutor;
3.7× on 222k, capped by slowest single Leiden (~60s). Intra-Leiden
parallelism would break result parity (leidenalg/igraph determinism).
See ``memory/feedback_rust_speedup_assumption.md``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# --- Process-pool worker state -----------------------------------------------
# Globals live in each worker process; main process never reads these.
_WORKER_GRAPH = None


def _set_process_priority(level: str) -> None:
    """Best-effort process-priority lowering so Leiden workers don't
    starve the foreground (video / browser) while they burn CPU.

    Pure stdlib — no psutil dep. ``level`` ∈ {"below_normal", "idle"}.
    Silently ignores platform errors.
    """
    import sys
    try:
        if sys.platform == "win32":
            import ctypes
            PRIORITY = {
                "below_normal": 0x00004000,
                "idle":         0x00000040,
            }
            code = PRIORITY.get(level)
            if code is None:
                return
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, code)
        else:
            import os as _os
            nice_delta = {"below_normal": 10, "idle": 19}.get(level, 0)
            if nice_delta:
                _os.nice(nice_delta)
    except Exception:
        pass


def _leiden_worker_init(graph_pickle: bytes, priority: str | None = None) -> None:
    """Deserialize the shared CSR graph once per worker, optionally lower
    process priority so the host stays responsive (v2-P9.1)."""
    import pickle
    global _WORKER_GRAPH
    _WORKER_GRAPH = pickle.loads(graph_pickle)
    if priority:
        _set_process_priority(priority)


def _leiden_worker(args: tuple) -> tuple:
    """Run a single Leiden in a worker. Returns ``(resolution, labels)``.

    Module-level (not closure) so Windows spawn-based ProcessPoolExecutor
    can import it. Reads ``_WORKER_GRAPH`` set up by
    ``_leiden_worker_init``.
    """
    resolution, seed, n_iterations, leiden_flavor = args
    import scanpy as sc
    import anndata as ad
    import scipy.sparse as sp

    G = _WORKER_GRAPH
    if G is None:
        raise RuntimeError("_WORKER_GRAPH not initialized — missing initializer?")
    n = G.shape[0]
    a = ad.AnnData(X=sp.csr_matrix((n, 1)))
    sc.tl.leiden(
        a, resolution=resolution, key_added="leiden",
        adjacency=G, flavor=leiden_flavor, n_iterations=n_iterations,
        directed=False, random_state=seed,
    )
    return resolution, a.obs["leiden"].astype(int).to_numpy()


def _default_max_workers(n_tasks: int, reserve_cores: int = 4) -> int:
    """Worker count default: reserve ``reserve_cores`` cores for OS /
    foreground apps so long Leiden sweeps don't freeze the host.
    """
    import os
    cpu = os.cpu_count() or 1
    return max(1, min(n_tasks, cpu - reserve_cores))


def _leiden_sweep(
    G,
    resolutions: list[float],
    *,
    seed: int = 0,
    n_iterations: int = 2,
    leiden_flavor: Literal["igraph", "leidenalg"] = "igraph",
    max_workers: int | None = None,
    worker_priority: str | None = "below_normal",
) -> dict[float, np.ndarray]:
    """Run Leiden at each resolution in parallel. Results are bit-identical
    to running ``sc.tl.leiden`` sequentially with the same seed —
    leidenalg is deterministic under a fixed ``random_state``.

    For ``len(resolutions) == 1`` we skip the process pool overhead and
    run in-process.

    ``max_workers=None`` → reserve 4 cores for OS / foreground apps (v2-P9.1).
    ``worker_priority="below_normal"`` (default) lowers each worker's OS
    priority so the host stays responsive while Leiden saturates CPU.
    Pass ``None`` to keep NORMAL priority.
    """
    import scanpy as sc
    import anndata as ad
    import scipy.sparse as sp

    if len(resolutions) == 0:
        return {}

    if len(resolutions) == 1:
        r = resolutions[0]
        n = G.shape[0]
        a = ad.AnnData(X=sp.csr_matrix((n, 1)))
        sc.tl.leiden(
            a, resolution=r, key_added="leiden",
            adjacency=G, flavor=leiden_flavor, n_iterations=n_iterations,
            directed=False, random_state=seed,
        )
        return {r: a.obs["leiden"].astype(int).to_numpy()}

    import pickle
    from concurrent.futures import ProcessPoolExecutor

    if max_workers is None:
        max_workers = _default_max_workers(len(resolutions))
    else:
        max_workers = max(1, min(len(resolutions), max_workers))

    graph_pickle = pickle.dumps(G)
    args_list = [(r, seed, n_iterations, leiden_flavor) for r in resolutions]
    out: dict[float, np.ndarray] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_leiden_worker_init,
        initargs=(graph_pickle, worker_priority),
    ) as ex:
        for r, lbl in ex.map(_leiden_worker, args_list):
            out[r] = lbl
    return out


def mean_conductance(G, labels: np.ndarray) -> float:
    """Size-weighted mean cluster conductance on sparse CSR graph G.

    conductance(C) = cut(C, V\\C) / min(vol(C), vol(V\\C))
    where vol(C) = Σ_{i∈C} deg(i). Lower values = tighter communities.
    Range [0, 1]. Singleton clusters are skipped (undefined boundary).

    O(nnz) — no subsampling. Deterministic given the graph + labels.
    Kept post-v2-P12 as a diagnostic primitive (CHAMP itself uses
    Newman/CPM modularity coefficients, not conductance).
    """
    labels = np.asarray(labels)
    deg = np.asarray(G.sum(axis=1)).ravel()
    total_vol = float(deg.sum())
    if total_vol == 0.0:
        return 1.0
    uniq = np.unique(labels)
    conds: list[float] = []
    sizes: list[int] = []
    for c in uniq:
        mask = labels == c
        n_c = int(mask.sum())
        if n_c < 2:
            continue
        vol_c = float(deg[mask].sum())
        vol_rest = total_vol - vol_c
        if vol_c == 0.0 or vol_rest == 0.0:
            continue
        G_rows = G[mask]
        row_sum = float(G_rows.sum())
        within = float(G_rows[:, mask].sum())
        cross = row_sum - within
        conds.append(cross / min(vol_c, vol_rest))
        sizes.append(n_c)
    if not conds:
        return 1.0
    weights = np.asarray(sizes, dtype=np.float64)
    return float(np.average(conds, weights=weights))


def auto_resolution(adata, method: str, conn, cfg) -> tuple[np.ndarray, float]:
    """Pick a Leiden resolution via CHAMP (Weir et al. 2017).

    Returns ``(labels, picked_gamma)``. CHAMP is the only optimizer
    since v2-P12 — see module docstring for the rationale and the
    config docstring for tuning knobs (``cfg.champ_*``).
    """
    from .champ import optimize_resolution_champ

    # Expose the route's connectivities as the scanpy-default key so
    # downstream consumers (plotting, scIB) see them.
    adata.obsp["connectivities"] = conn
    adata.uns["neighbors"] = {
        "params": {"method": "umap"},
        "connectivities_key": "connectivities",
    }

    df = optimize_resolution_champ(
        adata,
        method=method,
        conn=conn,
        n_partitions=int(cfg.champ_n_partitions),
        gamma_min=float(cfg.champ_gamma_min),
        gamma_max=float(cfg.champ_gamma_max),
        modularity=str(cfg.champ_modularity),
        width_metric=str(cfg.champ_width_metric),
        seed=0,
        leiden_flavor="igraph",
        leiden_n_iterations=cfg.leiden_n_iterations,
        max_workers=cfg.max_leiden_workers,
        worker_priority=cfg.leiden_worker_priority,
        verbose=True,
    )

    # Persist the full hull table for plotting / downstream inspection.
    adata.uns[f"champ_curve_{method}"] = {
        "origin_resolution": df["origin_resolution"].tolist(),
        "a":                 df["a"].tolist(),
        "b":                 df["b"].tolist(),
        "n_clusters":        df["n_clusters"].tolist(),
        "on_hull":           df["on_hull"].tolist(),
        "gamma_lo":          df["gamma_lo"].tolist(),
        "gamma_hi":          df["gamma_hi"].tolist(),
        "gamma_range":       df["gamma_range"].tolist(),
        "is_picked":         df["is_picked"].tolist(),
        "modularity":        cfg.champ_modularity,
        "width_metric":      cfg.champ_width_metric,
    }

    row = df.loc[df["is_picked"]].iloc[0]
    best_r = float(row["origin_resolution"])
    labels = adata.uns.pop(f"_champ_{method}_chosen_labels")
    print(f"         [{method}] [champ] picked γ={best_r:.3f} "
          f"(k={int(row['n_clusters'])}, admissible "
          f"γ∈[{row['gamma_lo']:.3f}, {row['gamma_hi']:.3f}])")
    return labels.astype(int), best_r


def plot_champ_curve(curve: pd.DataFrame, out_path, *, title: str | None = None) -> None:
    """Plot the CHAMP partition cloud + upper hull + picked partition.

    Two panels:
      * (b, a) plane with hull edges + crossover lines + picked vertex
      * γ-range bars per hull partition (admissible interval), picked highlighted
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    on_hull = curve["on_hull"].to_numpy().astype(bool)
    is_picked = curve["is_picked"].to_numpy().astype(bool)
    a = curve["a"].to_numpy()
    b = curve["b"].to_numpy()
    k = curve["n_clusters"].to_numpy()
    gamma_lo = curve["gamma_lo"].to_numpy()
    gamma_hi = curve["gamma_hi"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: (b, a) cloud + hull
    ax1.scatter(b[~on_hull], a[~on_hull], s=18, c="lightgrey",
                label="dominated", zorder=2)
    if on_hull.any():
        # Hull in ascending b order
        order = np.argsort(b[on_hull])
        hb, ha, hk = b[on_hull][order], a[on_hull][order], k[on_hull][order]
        ax1.plot(hb, ha, "-", color="steelblue", linewidth=1.2,
                 alpha=0.7, label="upper hull", zorder=3)
        ax1.scatter(hb, ha, s=36, c="steelblue", edgecolors="white",
                    linewidths=0.7, zorder=4)
        for x_, y_, k_ in zip(hb, ha, hk):
            ax1.annotate(f"k={k_}", (x_, y_), xytext=(4, 4),
                         textcoords="offset points", fontsize=8,
                         color="steelblue")
    if is_picked.any():
        ax1.scatter(b[is_picked], a[is_picked], s=140, marker="*",
                    c="crimson", edgecolors="black", linewidths=0.6,
                    label="picked (widest γ-range)", zorder=5)
    ax1.set_xlabel("b  (modularity slope coefficient)")
    ax1.set_ylabel("a  (modularity intercept)")
    ax1.set_title("CHAMP — Q(γ; P) = a − γ·b plane")
    ax1.legend(loc="best", fontsize=8, framealpha=0.85)
    ax1.grid(alpha=0.25)

    # Panel 2: γ-range bars per hull partition
    if on_hull.any():
        hull_idx_arr = np.where(on_hull)[0]
        order = np.argsort(b[hull_idx_arr])
        hull_idx_arr = hull_idx_arr[order]
        positions = np.arange(len(hull_idx_arr))
        for pos, idx in zip(positions, hull_idx_arr):
            color = "crimson" if is_picked[idx] else "steelblue"
            ax2.barh(pos, gamma_hi[idx] - gamma_lo[idx],
                     left=gamma_lo[idx], color=color, alpha=0.75,
                     edgecolor="black", linewidth=0.4)
        ax2.set_yticks(positions)
        ax2.set_yticklabels([f"k={k[i]}" for i in hull_idx_arr], fontsize=9)
        ax2.set_xlabel("admissible γ-range")
        ax2.set_title("Per-hull-partition γ-range  (★ = picked)")
        ax2.grid(axis="x", alpha=0.25)

    if title:
        fig.suptitle(title, fontsize=12, y=1.005)
    plt.tight_layout()
    # Dual-format save: PDF (Illustrator-editable) + PNG (preview).
    # The (b, a) scatter is only ~30 points so no rasterization needed
    # — entire figure is already vector-friendly.
    from ..plotting.comparison import _save_dual
    _save_dual(fig, out_path, dpi_pdf=300, dpi_png=150)

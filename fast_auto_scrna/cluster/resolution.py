"""Graph-based silhouette resolution optimizer.

Method: for each candidate Leiden resolution, compute silhouette over
subsampled cells using the neighbor-graph connectivity as the distance
source (d = 1 - connectivity). Evaluates clustering the way Leiden
sees it (graph topology) rather than going back to Euclidean on PCA.

Ported from v1 ``scatlas_pipeline/silhouette.py`` at V2-P2, plus the
``auto_resolution`` driver carved out of v1 ``pipeline._leiden_auto_resolution``.

Uses sklearn silhouette today. Milestone GS-3 swaps in the Rust
``silhouette_precomputed`` kernel via ``_silhouette_impl``; keep the
kernel call in one place so the swap is a one-line change.

GS-4 (2026-04-24): resolution sweep now parallelized across resolutions
via ProcessPoolExecutor (``_leiden_sweep``). Rust Leiden port was
microbenched and rejected — Python MP gives 3.7x on 222k (sequential
244.80s → parallel 66.01s), capped by the slowest single Leiden (~60s).
leidenalg's C++ igraph backend is hard to beat on single-run speed, and
intra-Leiden parallelism would break result parity. See
memory/feedback_rust_speedup_assumption.md.
"""
from __future__ import annotations

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
    process priority so the host stays responsive (v2-P9.1).

    Called by ``ProcessPoolExecutor(initializer=...)`` so the ~100 MB
    pickle payload is paid once per worker rather than once per call.
    """
    import pickle
    global _WORKER_GRAPH
    _WORKER_GRAPH = pickle.loads(graph_pickle)
    if priority:
        _set_process_priority(priority)


def _leiden_worker(args: tuple) -> tuple:
    """Run a single Leiden in a worker. Returns ``(resolution, labels)``.

    Module-level (not a closure) so Windows spawn-based
    ProcessPoolExecutor can import it. Reads ``_WORKER_GRAPH`` set up by
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
    to running ``sc.tl.leiden`` sequentially with the same seed — leidenalg
    is deterministic under a fixed ``random_state``.

    For ``len(resolutions) == 1`` we skip the process pool overhead and run
    in-process.

    ``max_workers=None`` → reserve 4 cores for OS / foreground apps (v2-P9.1).
    ``worker_priority="below_normal"`` (default) lowers each worker's OS
    priority so the host stays responsive while Leiden saturates CPU.
    Pass ``None`` to keep NORMAL priority (slightly faster but freezes
    video playback / browsers on heavily-loaded boxes).
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


def _silhouette_impl(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Single-shot silhouette on a precomputed distance matrix.

    Dispatches to ``fast_auto_scrna._native.silhouette.silhouette_precomputed``
    when the GS-3 kernel lands; otherwise falls back to sklearn.
    """
    try:
        from fast_auto_scrna._native import silhouette as _native_sil  # type: ignore
        if hasattr(_native_sil, "silhouette_precomputed"):
            return float(_native_sil.silhouette_precomputed(
                np.ascontiguousarray(distance_matrix, dtype=np.float32),
                np.ascontiguousarray(labels, dtype=np.int32),
            ))
    except ImportError:
        pass
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(distance_matrix, labels, metric="precomputed"))


def _stratified_sample(strata: np.ndarray, n_total: int, rng) -> np.ndarray:
    """Sample n_total cells preserving per-class proportion; ≥ 10 per class
    if that class has ≥ 10 cells."""
    classes, counts = np.unique(strata, return_counts=True)
    proportions = counts / counts.sum()
    picks = []
    for c, prop, cnt in zip(classes, proportions, counts):
        class_idx = np.where(strata == c)[0]
        n_want = max(10, int(round(n_total * prop)))
        n_want = min(n_want, cnt)
        picks.append(rng.choice(class_idx, size=n_want, replace=False))
    return np.concatenate(picks)


def optimize_resolution_graph_silhouette(
    adata,
    *,
    method: str,
    conn=None,
    neighbors_key: str | None = None,
    resolutions: list[float] | None = None,
    n_subsample: int = 1000,
    n_iter: int = 100,
    stratify_key: str | None = None,
    seed: int = 0,
    leiden_flavor: Literal["igraph", "leidenalg"] = "igraph",
    leiden_n_iterations: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """For each candidate resolution, compute mean±sd graph silhouette.

    Returns a DataFrame with columns ``resolution``, ``mean_silhouette``,
    ``sd_silhouette``, ``n_clusters``.

    Side effect: writes ``adata.obs[f"leiden_{method}_r{res:.2f}"]`` for
    each resolution.
    """
    import scanpy as sc

    if conn is None:
        graph_key = f"{method}_connectivities"
        if graph_key not in adata.obsp:
            raise KeyError(
                f"Missing {graph_key!r} in adata.obsp. Build the neighbor graph "
                f"first, or pass conn= directly."
            )
        G = adata.obsp[graph_key].tocsr()
    else:
        import scipy.sparse as _sp
        G = _sp.csr_matrix(conn)

    if resolutions is None:
        resolutions = list(np.round(np.arange(0.1, 1.55, 0.05), 2))

    # GS-4: parallelize across resolutions. Each Leiden is CPU-bound
    # (~49 s/call at 222 k), embarrassingly parallel across resolutions,
    # and deterministic under fixed seed — so parallel output is bit-
    # identical to sequential.
    labels_per_res = _leiden_sweep(
        G, list(resolutions),
        seed=seed,
        n_iterations=leiden_n_iterations,
        leiden_flavor=leiden_flavor,
    )
    # Write labels back into adata.obs in scanpy's canonical form
    # (categorical of string-typed integer labels).
    try:
        from natsort import natsorted
    except ImportError:
        natsorted = sorted  # type: ignore
    for r in resolutions:
        key = f"leiden_{method}_r{r:.2f}"
        lbl = labels_per_res[r]
        str_lbl = lbl.astype(str)
        cats = natsorted(np.unique(str_lbl).tolist())
        adata.obs[key] = pd.Categorical(str_lbl, categories=cats)
        if verbose:
            k = len(np.unique(lbl))
            print(f"    [silhouette] leiden r={r:.2f} → {k} clusters")

    rng = np.random.default_rng(seed)
    strata = (
        adata.obs[stratify_key].astype(str).to_numpy()
        if stratify_key is not None else None
    )
    scores: dict[float, list[float]] = {r: [] for r in resolutions}
    cluster_counts: dict[float, list[int]] = {r: [] for r in resolutions}

    N = adata.n_obs
    for _it in range(n_iter):
        if strata is not None:
            idx = _stratified_sample(strata, n_subsample, rng)
        else:
            idx = rng.choice(N, size=min(n_subsample, N), replace=False)

        sub = G[idx][:, idx].toarray().astype(np.float32)
        dist = 1.0 - sub
        np.fill_diagonal(dist, 0.0)

        for r in resolutions:
            lbl = labels_per_res[r][idx]
            uniq = np.unique(lbl)
            cluster_counts[r].append(len(uniq))
            if len(uniq) < 2:
                scores[r].append(0.0)
                continue
            scores[r].append(_silhouette_impl(dist, lbl))

    return pd.DataFrame({
        "resolution":      resolutions,
        "mean_silhouette": [float(np.mean(scores[r])) for r in resolutions],
        "sd_silhouette":   [float(np.std(scores[r]))  for r in resolutions],
        "n_clusters":      [int(np.median(cluster_counts[r])) for r in resolutions],
    })


def pick_best_resolution(
    curve: pd.DataFrame,
    *,
    k_lo: int | None = None,
    k_hi: int | None = None,
    metric: str = "mean_silhouette",
    direction: Literal["max", "min"] = "max",
) -> float:
    """Pick resolution by best ``metric`` value within optional k bounds.

    ``direction="max"`` for silhouette (higher = better);
    ``direction="min"`` for conductance (lower = better).
    """
    eligible = curve.copy()
    if k_lo is not None:
        eligible = eligible[eligible["n_clusters"] >= k_lo]
    if k_hi is not None:
        eligible = eligible[eligible["n_clusters"] <= k_hi]
    if len(eligible) == 0:
        raise ValueError(
            f"No resolution in curve satisfies k ∈ [{k_lo}, {k_hi}]. "
            f"n_clusters observed: {curve['n_clusters'].tolist()}"
        )
    idx = (eligible[metric].idxmax() if direction == "max"
           else eligible[metric].idxmin())
    return float(eligible.loc[idx, "resolution"])


def mean_conductance(G, labels: np.ndarray) -> float:
    """Size-weighted mean cluster conductance on sparse CSR graph G.

    conductance(C) = cut(C, V\\C) / min(vol(C), vol(V\\C))
    where vol(C) = Σ_{i∈C} deg(i). Lower values = tighter communities.
    Range [0, 1]. Singleton clusters are skipped (undefined boundary).

    O(nnz) — no subsampling. Deterministic given the graph + labels.
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
        # cross-boundary weight = (sum of G[mask, ~mask]). Compute as row-sum
        # inside C minus within-C sum to keep one CSR slice.
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


def optimize_resolution_conductance(
    adata,
    *,
    method: str,
    conn=None,
    resolutions: list[float] | None = None,
    seed: int = 0,
    leiden_flavor: Literal["igraph", "leidenalg"] = "igraph",
    leiden_n_iterations: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Leiden sweep + size-weighted mean conductance per resolution.

    Fast, deterministic, no subsampling. Lower conductance = tighter
    communities. Replaces the noise-dominated 1-connectivity silhouette
    on sparse kNN subsamples (see benchmarks/diagnose_silhouette.py).

    Returns DataFrame with columns ``resolution``, ``conductance``,
    ``n_clusters``. Side effect: writes
    ``adata.obs[f"leiden_{method}_r{res:.2f}"]`` for each resolution.
    """
    if conn is None:
        graph_key = f"{method}_connectivities"
        if graph_key not in adata.obsp:
            raise KeyError(
                f"Missing {graph_key!r} in adata.obsp. Build the neighbor graph "
                f"first, or pass conn= directly."
            )
        G = adata.obsp[graph_key].tocsr()
    else:
        import scipy.sparse as _sp
        G = _sp.csr_matrix(conn)

    if resolutions is None:
        resolutions = list(np.round(np.arange(0.1, 1.55, 0.05), 2))

    labels_per_res = _leiden_sweep(
        G, list(resolutions),
        seed=seed,
        n_iterations=leiden_n_iterations,
        leiden_flavor=leiden_flavor,
    )

    try:
        from natsort import natsorted
    except ImportError:
        natsorted = sorted  # type: ignore
    for r in resolutions:
        key = f"leiden_{method}_r{r:.2f}"
        lbl = labels_per_res[r]
        str_lbl = lbl.astype(str)
        cats = natsorted(np.unique(str_lbl).tolist())
        adata.obs[key] = pd.Categorical(str_lbl, categories=cats)

    conds: list[float] = []
    ks: list[int] = []
    for r in resolutions:
        lbl = labels_per_res[r]
        c = mean_conductance(G, lbl)
        k = int(len(np.unique(lbl)))
        conds.append(c)
        ks.append(k)
        if verbose:
            print(f"    [conductance] leiden r={r:.2f} → k={k}  cond={c:.4f}")

    return pd.DataFrame({
        "resolution":  resolutions,
        "conductance": conds,
        "n_clusters":  ks,
    })


def perpendicular_elbow(y: np.ndarray) -> int:
    """PCA-style global Kneedle: index farthest from the
    ``[(0, y[0]), (L-1, y[-1])]`` secant (absolute perpendicular distance).

    Mirrors the Rust ``perpendicular_elbow`` in ``kernels/src/pca.rs`` used
    for PCA scree selection, adapted to 0-based index. Finds the GLOBAL
    max deviation — on multi-step conductance curves this tends to hit
    the middle step rather than the first plateau entry. Kept as a
    fallback detector.
    """
    y = np.asarray(y, dtype=np.float64)
    L = y.shape[0]
    if L < 3:
        return 0
    x1, y1 = 0.0, float(y[0])
    x2, y2 = float(L - 1), float(y[-1])
    dx = x2 - x1
    dy = y2 - y1
    line_norm = max(float(np.hypot(dx, dy)), 1e-20)
    px = np.arange(L, dtype=np.float64)
    num = np.abs(dy * px - dx * y + x2 * y1 - y2 * x1)
    dist = num / line_norm
    dist[0] = 0.0
    dist[-1] = 0.0
    return int(np.argmax(dist))


def first_plateau_after_rise(
    y: np.ndarray,
    *,
    window: int = 5,
    min_rise_ratio: float = 0.10,
    low_slope_ratio: float = 0.25,
) -> int:
    """Detect the first plateau entry after the initial rapid-rise segment.

    Scans left-to-right. Returns the first index ``i`` such that
      (a) ``y[i] - y[0] >= min_rise_ratio × (max(y) - min(y))``  —
          we have escaped the initial low plateau, AND
      (b) the local slope at ``i`` (rolling window of ``window`` points)
          has dropped below ``low_slope_ratio × max_slope_seen``  —
          we have entered a flatter region.

    Fallback: ``argmax(slopes)`` if no index satisfies both.

    This matches the user's directive "快速上升到平台的第一个点" — the
    entry of the first plateau after the initial rise, not the globally
    most-bowed point (perpendicular_elbow fails here when the curve has
    later bigger jumps that pull the global secant askew).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    if n < window:
        return 0
    slopes = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo >= 2:
            slopes[i] = (y[hi - 1] - y[lo]) / (hi - 1 - lo)
    rng = float(y.max() - y.min())
    if rng <= 0.0:
        return 0
    min_rise = min_rise_ratio * rng
    max_seen = 0.0
    for i in range(n):
        if slopes[i] > max_seen:
            max_seen = float(slopes[i])
        if (y[i] - y[0]) >= min_rise and max_seen > 0.0 \
           and slopes[i] < low_slope_ratio * max_seen:
            return i
    return int(np.argmax(slopes))


_DETECTORS = {
    "first_plateau": first_plateau_after_rise,
    "perp_elbow":    lambda y: perpendicular_elbow(np.asarray(y)),
}


def optimize_resolution_knee(
    adata,
    *,
    method: str,
    conn=None,
    resolutions: list[float] | None = None,
    offset_steps: int = 3,
    detector: Literal["first_plateau", "perp_elbow"] = "first_plateau",
    two_stage: bool = True,
    fine_step: float = 0.01,
    fine_half_width: float = 0.05,
    seed: int = 0,
    leiden_flavor: Literal["igraph", "leidenalg"] = "igraph",
    leiden_n_iterations: int = 2,
    max_workers: int | None = None,
    worker_priority: str | None = "below_normal",
    verbose: bool = True,
) -> pd.DataFrame:
    """Two-stage knee picker on the conductance-vs-resolution curve.

    Stage 1 — coarse: sweep ``resolutions`` (default 10 points spanning
      0.05..1.00) and detect an approximate knee via ``detector``.
    Stage 2 — fine:   sweep ``[knee_r - fine_half_width,
      knee_r + fine_half_width]`` at ``fine_step`` step, re-detect knee
      on the fine curve, pick ``fine_knee_idx + offset_steps``.

    Cost: ~2-stage sweep is O(coarse_n + fine_n) Leidens per route, vs
    O(~150) for a full 0.01 sweep. On 222k atlas this drops wall from
    ~70 min to ~15 min while hitting the same picked resolution region.

    Set ``two_stage=False`` to disable the refinement and use a single
    sweep (previous behavior).

    Returns DataFrame with columns ``resolution``, ``conductance``,
    ``n_clusters``, ``is_knee`` (bool), ``is_picked`` (bool), ``stage``
    (``"coarse"`` or ``"fine"``). The picker's final decision is on the
    fine stage when ``two_stage=True``.
    """
    if conn is None:
        graph_key = f"{method}_connectivities"
        if graph_key not in adata.obsp:
            raise KeyError(
                f"Missing {graph_key!r} in adata.obsp. Build the neighbor graph "
                f"first, or pass conn= directly."
            )
        G = adata.obsp[graph_key].tocsr()
    else:
        import scipy.sparse as _sp
        G = _sp.csr_matrix(conn)

    if resolutions is None:
        if two_stage:
            resolutions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                           0.40, 0.50, 0.70, 1.00]
        else:
            resolutions = [round(r, 2) for r in np.arange(0.01, 1.51, 0.01)]

    if detector not in _DETECTORS:
        raise ValueError(
            f"detector={detector!r} — must be one of {list(_DETECTORS)}"
        )

    try:
        from natsort import natsorted
    except ImportError:
        natsorted = sorted  # type: ignore

    def _write_obs(labels_dict):
        for r, lbl in labels_dict.items():
            key = f"leiden_{method}_r{r:.2f}"
            str_lbl = lbl.astype(str)
            cats = natsorted(np.unique(str_lbl).tolist())
            adata.obs[key] = pd.Categorical(str_lbl, categories=cats)

    def _curve(labels_dict, rs):
        conds_, ks_ = [], []
        for r in rs:
            lbl = labels_dict[r]
            conds_.append(mean_conductance(G, lbl))
            ks_.append(int(len(np.unique(lbl))))
        return conds_, ks_

    # --- Stage 1: coarse ---
    coarse_rs = sorted(set(round(float(r), 2) for r in resolutions))
    if verbose:
        print(f"    [knee/{detector}] stage-1 coarse: "
              f"{len(coarse_rs)} resolutions {coarse_rs[0]}..{coarse_rs[-1]}")
    coarse_labels = _leiden_sweep(
        G, list(coarse_rs), seed=seed,
        n_iterations=leiden_n_iterations, leiden_flavor=leiden_flavor,
        max_workers=max_workers, worker_priority=worker_priority,
    )
    _write_obs(coarse_labels)
    coarse_conds, coarse_ks = _curve(coarse_labels, coarse_rs)
    coarse_conds_arr = np.asarray(coarse_conds, dtype=np.float64)
    coarse_knee_idx = _DETECTORS[detector](coarse_conds_arr)
    coarse_knee_r = float(coarse_rs[coarse_knee_idx])
    if verbose:
        print(f"    [knee/{detector}] stage-1 knee at r={coarse_knee_r:.2f} "
              f"(k={coarse_ks[coarse_knee_idx]}, "
              f"cond={coarse_conds[coarse_knee_idx]:.4f})")

    if not two_stage:
        picked_idx = min(coarse_knee_idx + offset_steps, len(coarse_rs) - 1)
        picked_r = float(coarse_rs[picked_idx])
        df = pd.DataFrame({
            "resolution":  coarse_rs,
            "conductance": coarse_conds,
            "n_clusters":  coarse_ks,
            "stage":       ["coarse"] * len(coarse_rs),
        })
        df["is_knee"]   = df["resolution"] == coarse_knee_r
        df["is_picked"] = df["resolution"] == picked_r
        if verbose:
            print(f"    [knee/{detector}] picked r={picked_r:.2f} "
                  f"(k={coarse_ks[picked_idx]}) = knee + {offset_steps} steps "
                  f"(single-stage)")
        return df

    # --- Stage 2: fine around coarse knee ---
    # Extend +offset_steps * fine_step on the right so the picker has
    # enough room past the refined knee without clamping.
    fine_lo = max(0.01, coarse_knee_r - fine_half_width)
    fine_hi = coarse_knee_r + fine_half_width + offset_steps * fine_step
    n_fine = int(round((fine_hi - fine_lo) / fine_step)) + 1
    fine_rs = sorted({round(fine_lo + i * fine_step, 2) for i in range(n_fine)})
    # skip resolutions already computed in coarse
    fine_rs_new = [r for r in fine_rs if r not in coarse_labels]
    if verbose:
        print(f"    [knee/{detector}] stage-2 fine: {len(fine_rs)} resolutions "
              f"{fine_rs[0]}..{fine_rs[-1]} (step {fine_step}), "
              f"{len(fine_rs_new)} new Leidens")
    fine_labels_new = _leiden_sweep(
        G, fine_rs_new, seed=seed,
        n_iterations=leiden_n_iterations, leiden_flavor=leiden_flavor,
        max_workers=max_workers, worker_priority=worker_priority,
    ) if fine_rs_new else {}
    fine_labels = {**{r: coarse_labels[r] for r in fine_rs if r in coarse_labels},
                   **fine_labels_new}
    _write_obs(fine_labels_new)

    fine_conds, fine_ks = _curve(fine_labels, fine_rs)
    fine_knee_idx = _DETECTORS[detector](np.asarray(fine_conds, dtype=np.float64))
    fine_knee_r = float(fine_rs[fine_knee_idx])
    picked_idx = min(fine_knee_idx + offset_steps, len(fine_rs) - 1)
    picked_r = float(fine_rs[picked_idx])
    if verbose:
        print(f"    [knee/{detector}] stage-2 knee at r={fine_knee_r:.2f} "
              f"(k={fine_ks[fine_knee_idx]}, cond={fine_conds[fine_knee_idx]:.4f})")
        print(f"    [knee/{detector}] picked r={picked_r:.2f} "
              f"(k={fine_ks[picked_idx]}, cond={fine_conds[picked_idx]:.4f}) "
              f"= fine knee + {offset_steps} steps")

    # Build combined frame: coarse (dropping overlap) + fine, sorted.
    coarse_only = [r for r in coarse_rs if r not in fine_labels]
    rows = []
    for r in coarse_only:
        rows.append({"resolution": r, "stage": "coarse"})
    for r in fine_rs:
        rows.append({"resolution": r, "stage": "fine"})
    df = pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)
    all_labels = {**coarse_labels, **fine_labels}
    df["conductance"] = [mean_conductance(G, all_labels[r]) for r in df["resolution"]]
    df["n_clusters"]  = [int(len(np.unique(all_labels[r]))) for r in df["resolution"]]
    df["is_knee"]   = df["resolution"] == fine_knee_r
    df["is_picked"] = df["resolution"] == picked_r
    return df


def plot_knee_curve(
    curve: pd.DataFrame,
    out_path,
    *,
    title: str | None = None,
) -> None:
    """Plot conductance vs resolution with knee + picked markers.

    If the curve has a ``stage`` column, coarse vs fine points are drawn
    with different markers so the two-stage structure is visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    x = curve["resolution"].to_numpy()
    y = curve["conductance"].to_numpy()
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    if "stage" in curve.columns:
        coarse_mask = (curve["stage"] == "coarse").to_numpy()
        fine_mask   = (curve["stage"] == "fine").to_numpy()
        ax1.plot(x, y, "-", color="darkorange", linewidth=1.1, alpha=0.6)
        if coarse_mask.any():
            ax1.plot(x[coarse_mask], y[coarse_mask], "s", color="steelblue",
                     markersize=6, label="stage-1 coarse")
        if fine_mask.any():
            ax1.plot(x[fine_mask], y[fine_mask], "o", color="darkorange",
                     markersize=4, label="stage-2 fine")
    else:
        ax1.plot(x, y, "o-", color="darkorange", linewidth=1.1, alpha=0.9,
                 markersize=3, label="mean conductance")
    ax1.plot([x[0], x[-1]], [y[0], y[-1]], "--", color="grey",
             linewidth=0.7, alpha=0.6, label="secant")
    knee_row   = curve.loc[curve["is_knee"]]
    picked_row = curve.loc[curve["is_picked"]]
    if len(knee_row):
        r_k = float(knee_row.iloc[0]["resolution"])
        k_k = int(knee_row.iloc[0]["n_clusters"])
        ax1.axvline(r_k, linestyle=":", color="steelblue", linewidth=1.1,
                    label=f"knee r={r_k:.2f} (k={k_k})")
    if len(picked_row):
        r_p = float(picked_row.iloc[0]["resolution"])
        k_p = int(picked_row.iloc[0]["n_clusters"])
        ax1.axvline(r_p, linestyle="--", color="red", linewidth=1.3,
                    label=f"picked r={r_p:.2f} (k={k_p})")
    ax1.set_xlabel("leiden resolution")
    ax1.set_ylabel("mean conductance")
    ax1.set_title(title or "Conductance vs leiden resolution (knee picker)")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, curve["n_clusters"], "-", color="lightsteelblue",
             alpha=0.45, linewidth=0.8)
    ax2.set_ylabel("n_clusters", color="steelblue")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_conductance_curve(
    curve: pd.DataFrame,
    out_path,
    *,
    best_resolution: float | None = None,
    title: str | None = None,
    k_lo: int | None = None,
    k_hi: int | None = None,
) -> None:
    """Plot conductance vs resolution, n_clusters on twin axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(
        curve["resolution"], curve["conductance"],
        "o-", color="darkorange", label="mean conductance (lower=tighter)",
    )
    ax1.set_xlabel("leiden resolution")
    ax1.set_ylabel("conductance", color="darkorange")
    ax1.tick_params(axis="y", labelcolor="darkorange")

    ax2 = ax1.twinx()
    ax2.plot(
        curve["resolution"], curve["n_clusters"],
        "s--", color="steelblue", alpha=0.55, label="n_clusters",
    )
    ax2.set_ylabel("n clusters", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")

    if k_lo is not None or k_hi is not None:
        eligible = curve.copy()
        if k_lo is not None:
            eligible = eligible[eligible["n_clusters"] >= k_lo]
        if k_hi is not None:
            eligible = eligible[eligible["n_clusters"] <= k_hi]
        if len(eligible) > 0:
            ax1.axvspan(
                eligible["resolution"].min(), eligible["resolution"].max(),
                color="green", alpha=0.08,
                label=f"k∈[{k_lo},{k_hi}]",
            )

    if best_resolution is not None:
        row = curve.loc[(curve["resolution"] - best_resolution).abs().idxmin()]
        ax1.axvline(
            best_resolution, color="green", linestyle=":",
            label=f"best res={best_resolution:.2f} (k={int(row['n_clusters'])})",
        )

    ax1.legend(loc="upper right")
    ax1.set_title(title or "Mean cluster conductance vs leiden resolution")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_curve(
    curve: pd.DataFrame,
    out_path,
    *,
    best_resolution: float | None = None,
    title: str | None = None,
    k_lo: int | None = None,
    k_hi: int | None = None,
) -> None:
    """Plot mean±sd silhouette vs resolution with n_clusters on twin axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.errorbar(
        curve["resolution"], curve["mean_silhouette"],
        yerr=curve["sd_silhouette"], fmt="o-",
        color="steelblue", ecolor="lightsteelblue",
        capsize=3, label="mean ± sd silhouette",
    )
    ax1.set_xlabel("leiden resolution")
    ax1.set_ylabel("graph silhouette", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.axhline(0.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(
        curve["resolution"], curve["n_clusters"],
        "s--", color="coral", alpha=0.55, label="n_clusters",
    )
    ax2.set_ylabel("n clusters", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")

    if k_lo is not None or k_hi is not None:
        eligible = curve.copy()
        if k_lo is not None:
            eligible = eligible[eligible["n_clusters"] >= k_lo]
        if k_hi is not None:
            eligible = eligible[eligible["n_clusters"] <= k_hi]
        if len(eligible) > 0:
            ax1.axvspan(
                eligible["resolution"].min(), eligible["resolution"].max(),
                color="green", alpha=0.08,
                label=f"k∈[{k_lo},{k_hi}]",
            )

    if best_resolution is not None:
        row = curve.loc[(curve["resolution"] - best_resolution).abs().idxmin()]
        ax1.axvline(
            best_resolution, color="green", linestyle=":",
            label=f"best res={best_resolution:.2f} (k={int(row['n_clusters'])})",
        )

    ax1.legend(loc="upper left")
    ax1.set_title(title or "Graph silhouette vs leiden resolution")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def auto_resolution(adata, method: str, conn, cfg) -> tuple[np.ndarray, float]:
    """Pick a Leiden resolution using the configured optimizer.

    - ``cfg.resolution_optimizer == "knee"`` (default v2-P8): fine grid
      (``cfg.leiden_resolutions`` default ≈ 0.01..1.50 step 0.01), compute
      conductance per resolution, detect PCA-style perpendicular-line
      elbow on the curve, pick ``knee_idx + cfg.knee_offset_steps``. Biases
      toward finer clustering so the user's post-hoc marker-merge workflow
      can recover biology. NO k-range clip in this path.
    - ``cfg.resolution_optimizer == "conductance"``: argmin conductance
      within ``cfg.leiden_target_n`` clip. Legacy v2-P7 default. Replaced
      after pancreas audit showed argmin edge-picks on trajectory data.
    - ``cfg.resolution_optimizer == "graph_silhouette"`` (legacy): pick
      the one with highest mean silhouette on subsampled 1-connectivity.
    - ``cfg.resolution_optimizer == "target_n"``: smallest res giving
      ``n_clusters`` in ``cfg.leiden_target_n``. Legacy heuristic.

    Ported from v1 ``pipeline._leiden_auto_resolution``.
    """
    import scanpy as sc

    # Expose the route's connectivities as the scanpy-default key so subsequent
    # leiden calls see them.
    adata.obsp["connectivities"] = conn
    adata.uns["neighbors"] = {
        "params": {"method": "umap"},
        "connectivities_key": "connectivities",
    }

    if cfg.resolution_optimizer == "target_n":
        chosen: tuple[float, np.ndarray] | None = None
        for r in cfg.leiden_resolutions:
            key = f"_leiden_{method}_r{r}"
            sc.tl.leiden(
                adata, resolution=r, key_added=key,
                flavor="igraph", directed=False,
                n_iterations=cfg.leiden_n_iterations, random_state=0,
            )
            n = adata.obs[key].nunique()
            in_range = cfg.leiden_target_n[0] <= n <= cfg.leiden_target_n[1]
            print(f"         [{method}] leiden r={r} → {n} clusters"
                  f"{' ✓' if in_range else ''}")
            if in_range and chosen is None:
                chosen = (r, adata.obs[key].to_numpy())
        if chosen is None:
            r = cfg.leiden_resolutions[len(cfg.leiden_resolutions) // 2]
            print(f"         [{method}] [fallback] no res in target; r={r}")
            chosen = (r, adata.obs[f"_leiden_{method}_r{r}"].to_numpy())
        return chosen[1], chosen[0]

    if cfg.resolution_optimizer == "knee":
        offset           = int(getattr(cfg, "knee_offset_steps", 3))
        detector         = str(getattr(cfg, "knee_detector", "first_plateau"))
        two_stage        = bool(getattr(cfg, "knee_two_stage", True))
        fine_step        = float(getattr(cfg, "knee_fine_step", 0.01))
        fine_half_width  = float(getattr(cfg, "knee_fine_half_width", 0.05))
        max_workers      = getattr(cfg, "max_leiden_workers", None)
        worker_priority  = getattr(cfg, "leiden_worker_priority", "below_normal")
        curve = optimize_resolution_knee(
            adata,
            method=method,
            conn=conn,
            resolutions=cfg.leiden_resolutions,
            offset_steps=offset,
            detector=detector,
            two_stage=two_stage,
            fine_step=fine_step,
            fine_half_width=fine_half_width,
            seed=0,
            leiden_flavor="igraph",
            leiden_n_iterations=cfg.leiden_n_iterations,
            max_workers=max_workers,
            worker_priority=worker_priority,
            verbose=True,
        )
        adata.uns[f"knee_curve_{method}"] = {
            "resolution":  curve["resolution"].tolist(),
            "conductance": curve["conductance"].tolist(),
            "n_clusters":  curve["n_clusters"].tolist(),
            "is_knee":     curve["is_knee"].tolist(),
            "is_picked":   curve["is_picked"].tolist(),
            "stage":       curve["stage"].tolist() if "stage" in curve.columns
                           else ["single"] * len(curve),
        }
        row = curve.loc[curve["is_picked"]].iloc[0]
        best_r = float(row["resolution"])
        best_labels_key = f"leiden_{method}_r{best_r:.2f}"
        labels = adata.obs[best_labels_key].astype(int).to_numpy()
        print(f"         [{method}] [knee] picked r={best_r:.2f} "
              f"(k={int(row['n_clusters'])}, cond={row['conductance']:.4f}, "
              f"offset={offset})")
        return labels, best_r

    if cfg.resolution_optimizer == "conductance":
        curve = optimize_resolution_conductance(
            adata,
            method=method,
            conn=conn,
            resolutions=cfg.leiden_resolutions,
            seed=0,
            leiden_flavor="igraph",
            leiden_n_iterations=cfg.leiden_n_iterations,
            verbose=True,
        )
        adata.uns[f"conductance_curve_{method}"] = {
            "resolution":  curve["resolution"].tolist(),
            "conductance": curve["conductance"].tolist(),
            "n_clusters":  curve["n_clusters"].tolist(),
        }
        k_lo, k_hi = cfg.leiden_target_n
        try:
            best_r = pick_best_resolution(
                curve, k_lo=k_lo, k_hi=k_hi,
                metric="conductance", direction="min",
            )
        except ValueError:
            best_r = pick_best_resolution(
                curve, metric="conductance", direction="min",
            )
            print(f"         [{method}] [conductance] no res in "
                  f"k∈[{k_lo},{k_hi}]; unclipped best r={best_r}")

        best_labels_key = f"leiden_{method}_r{best_r:.2f}"
        labels = adata.obs[best_labels_key].astype(int).to_numpy()
        row = curve.loc[curve["resolution"] == best_r].iloc[0]
        print(f"         [{method}] [conductance] picked r={best_r:.2f} "
              f"(k={int(row['n_clusters'])}, cond={row['conductance']:.4f})")
        return labels, best_r

    if cfg.resolution_optimizer != "graph_silhouette":
        raise ValueError(
            f"resolution_optimizer={cfg.resolution_optimizer!r} — "
            f"must be 'knee', 'conductance', 'graph_silhouette', or 'target_n'"
        )

    # Stratify key reuses the sweep's own smallest-resolution Leiden output.
    # `optimize_resolution_graph_silhouette` runs Leiden at every resolution
    # in step 1 BEFORE consuming stratify_key in step 2, so the obs column
    # `leiden_{method}_r{r0:.2f}` is written by the time strata is read.
    # This removes a redundant Leiden run (one free ~15% Phase-2 speedup).
    stratify_key = None
    if cfg.silhouette_stratify:
        r0 = min(cfg.leiden_resolutions)
        stratify_key = f"leiden_{method}_r{r0:.2f}"

    curve = optimize_resolution_graph_silhouette(
        adata,
        method=method,
        conn=conn,
        neighbors_key=None,
        resolutions=cfg.leiden_resolutions,
        n_subsample=cfg.silhouette_n_subsample,
        n_iter=cfg.silhouette_n_iter,
        stratify_key=stratify_key,
        seed=0,
        leiden_flavor="igraph",
        leiden_n_iterations=cfg.leiden_n_iterations,
        verbose=True,
    )
    adata.uns[f"silhouette_curve_{method}"] = {
        "resolution":      curve["resolution"].tolist(),
        "mean_silhouette": curve["mean_silhouette"].tolist(),
        "sd_silhouette":   curve["sd_silhouette"].tolist(),
        "n_clusters":      curve["n_clusters"].tolist(),
    }

    k_lo, k_hi = cfg.leiden_target_n
    try:
        best_r = pick_best_resolution(curve, k_lo=k_lo, k_hi=k_hi)
    except ValueError:
        best_r = pick_best_resolution(curve)
        print(f"         [{method}] [silhouette] no res in k∈[{k_lo},{k_hi}]; "
              f"unclipped best r={best_r}")

    best_labels_key = f"leiden_{method}_r{best_r:.2f}"
    if best_labels_key not in adata.obs.columns:
        sc.tl.leiden(
            adata, resolution=best_r, key_added=best_labels_key,
            flavor="igraph", directed=False,
            n_iterations=cfg.leiden_n_iterations, random_state=0,
        )
    labels = adata.obs[best_labels_key].to_numpy()
    print(f"         [{method}] [silhouette] picked r={best_r:.2f} by "
          f"mean_silhouette (curve stored in uns)")
    return labels, best_r

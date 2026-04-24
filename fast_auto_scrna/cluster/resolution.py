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


def _leiden_worker_init(graph_pickle: bytes) -> None:
    """Deserialize the shared CSR graph once per worker.

    Called by ``ProcessPoolExecutor(initializer=...)`` so the ~100 MB
    pickle payload is paid once per worker rather than once per call.
    """
    import pickle
    global _WORKER_GRAPH
    _WORKER_GRAPH = pickle.loads(graph_pickle)


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


def _leiden_sweep(
    G,
    resolutions: list[float],
    *,
    seed: int = 0,
    n_iterations: int = 2,
    leiden_flavor: Literal["igraph", "leidenalg"] = "igraph",
    max_workers: int | None = None,
) -> dict[float, np.ndarray]:
    """Run Leiden at each resolution in parallel. Results are bit-identical
    to running ``sc.tl.leiden`` sequentially with the same seed — leidenalg
    is deterministic under a fixed ``random_state``.

    For ``len(resolutions) == 1`` we skip the process pool overhead and run
    in-process.
    """
    import os
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
        max_workers = min(len(resolutions), os.cpu_count() or 1)

    graph_pickle = pickle.dumps(G)
    args_list = [(r, seed, n_iterations, leiden_flavor) for r in resolutions]
    out: dict[float, np.ndarray] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_leiden_worker_init,
        initargs=(graph_pickle,),
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

    - ``cfg.resolution_optimizer == "conductance"`` (default v2-P7): run
      all resolutions, pick the one with MIN size-weighted mean cluster
      conductance on the full graph. Replaces graph_silhouette after the
      metric audit found silhouette(1 - connectivity) is noise-dominated
      on sparse kNN subsamples.
    - ``cfg.resolution_optimizer == "graph_silhouette"`` (legacy, kept
      for backward compat): pick the one with highest mean silhouette on
      subsampled distance = 1 - connectivity.
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
            f"must be 'conductance', 'graph_silhouette', or 'target_n'"
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

"""Graph-based silhouette resolution optimizer.

Method: for each candidate Leiden resolution, compute silhouette over
subsampled cells using the neighbor-graph connectivity as the distance
source (d = 1 - connectivity). This evaluates clustering the SAME way
Leiden sees it — graph topology — rather than going back to Euclidean
in the PCA embedding.

Works for any integration method whose neighbors graph is stored as
adata.obsp[f"{method}_connectivities"] (scanpy convention when
sc.pp.neighbors was called with key_added=method).

Uses sklearn silhouette for now. Task GS-3 will swap a Rust kernel in
via the _silhouette_impl hook; keep the kernel call in one place so the
swap is a one-line change.
"""
from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _silhouette_impl(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Single-shot silhouette on a precomputed distance matrix.

    Dispatches to scvalidate_rust.silhouette_precomputed when available
    (GS-4 wires this up). Falls back to sklearn.
    """
    try:
        import scvalidate_rust
        if hasattr(scvalidate_rust, "silhouette_precomputed"):
            return float(scvalidate_rust.silhouette_precomputed(
                np.ascontiguousarray(distance_matrix, dtype=np.float32),
                np.ascontiguousarray(labels, dtype=np.int32),
            ))
    except ImportError:
        pass
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(distance_matrix, labels, metric="precomputed"))


def _stratified_sample(strata: np.ndarray, n_total: int, rng) -> np.ndarray:
    """Sample n_total cells preserving per-class proportion; at least 10 per
    class if that class has ≥10 cells. Returns an array of N-scale indices.
    """
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

    Returns a DataFrame with columns:
        resolution, mean_silhouette, sd_silhouette, n_clusters

    Side effect: adds adata.obs[f"leiden_{method}_r{res:.2f}"] for each
    resolution (so the caller can pick one after seeing the curve).

    Parameters
    ----------
    conn
        If provided, use this CSR connectivity matrix directly and skip
        the ``adata.obsp[f"{method}_connectivities"]`` lookup.  Allows
        callers that manage the connectivities externally (e.g. pipeline
        routes that write into the generic ``obsp["connectivities"]``) to
        avoid registering a per-method key.
    neighbors_key
        Retained for API compatibility; ignored.  The optimizer now passes
        the graph to ``sc.tl.leiden`` via ``adjacency=`` directly, so no
        lookup through ``adata.uns[...]`` happens.

    Raises KeyError if ``conn`` is None and
    ``adata.obsp[f"{method}_connectivities"]`` is missing.
    """
    import scanpy as sc

    if conn is None:
        graph_key = f"{method}_connectivities"
        if graph_key not in adata.obsp:
            raise KeyError(
                f"Missing {graph_key!r} in adata.obsp. Build the neighbor graph "
                f"with sc.pp.neighbors(..., key_added={method!r}) first, "
                f"or pass conn= directly."
            )
        G = adata.obsp[graph_key].tocsr()
    else:
        import scipy.sparse as _sp
        G = _sp.csr_matrix(conn)

    if resolutions is None:
        resolutions = list(np.round(np.arange(0.1, 1.55, 0.05), 2))

    # 1) Run Leiden at each resolution; cache labels N-wide.
    #    Pass adjacency=G directly — bypasses scanpy's NeighborsView lookup,
    #    so we don't depend on adata.uns[method] being registered. Works for
    #    both flavor="igraph" and flavor="leidenalg" (scanpy _leiden.py L148).
    labels_per_res: dict[float, np.ndarray] = {}
    for r in resolutions:
        key = f"leiden_{method}_r{r:.2f}"
        sc.tl.leiden(
            adata, resolution=r, key_added=key,
            adjacency=G,
            flavor=leiden_flavor, n_iterations=leiden_n_iterations,
            directed=False, random_state=seed,
        )
        labels_per_res[r] = adata.obs[key].astype(int).to_numpy()
        if verbose:
            k = len(np.unique(labels_per_res[r]))
            print(f"    [silhouette] leiden r={r:.2f} → {k} clusters")

    # 2) Subsample iterations — compute silhouette for each resolution on the
    #    same sampled indices so iteration variance is shared across res.
    rng = np.random.default_rng(seed)
    strata = (
        adata.obs[stratify_key].astype(str).to_numpy()
        if stratify_key is not None else None
    )
    scores: dict[float, list[float]] = {r: [] for r in resolutions}
    cluster_counts: dict[float, list[int]] = {r: [] for r in resolutions}

    N = adata.n_obs
    for it in range(n_iter):
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


def plot_silhouette_curve(
    curve: pd.DataFrame,
    out_path,
    *,
    best_resolution: float | None = None,
    title: str | None = None,
    k_lo: int | None = None,
    k_hi: int | None = None,
) -> None:
    """Plot mean±sd silhouette vs resolution with n_clusters on twin axis.

    Saved as PNG at ``out_path``. Shades the eligible k-band (k_lo..k_hi)
    and highlights the picked resolution if provided.
    """
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


def pick_best_resolution(
    curve: pd.DataFrame,
    *,
    k_lo: int | None = None,
    k_hi: int | None = None,
) -> float:
    """Pick resolution with highest mean_silhouette.

    Optional k_lo/k_hi clips the eligible rows to those whose n_clusters
    is in [k_lo, k_hi]. Returns the resolution (float); raises if no row
    satisfies the bounds.
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
    row = eligible.loc[eligible["mean_silhouette"].idxmax()]
    return float(row["resolution"])

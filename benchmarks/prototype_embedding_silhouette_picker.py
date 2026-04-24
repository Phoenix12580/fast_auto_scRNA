"""Prototype: embedding-silhouette resolution picker, fine grid + CI.

Replaces the failed 1-connectivity silhouette with Euclidean silhouette
on the integrated PCA embedding — the signal that the user's previous
R workflow exposed (0.42 plateau 0.05-0.20, step-down thereafter).

On pancreas (1000 cells, 8 SubCellTypes), check:
  1. curve has visible peak/plateau (not monotonic)
  2. picker hits r ≈ 0.2-0.3 matching ground-truth ARI (0.65)
  3. conductance / current default would have picked r=0.05 (ARI 0.44)

Uses the GS-3 Rust silhouette_precomputed kernel for speed.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import os
import time
import pickle
import numpy as np
import pandas as pd
import anndata as ad
from concurrent.futures import ProcessPoolExecutor


H5AD       = "benchmarks/out/smoke_pancreas.h5ad"
REP        = "X_pca"        # pancreas single-batch, no harmony/bbknn
LABEL_COL  = "SubCellType"
RESOLUTIONS = [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
               0.25, 0.30, 0.40, 0.50, 0.70, 1.00]
N_SUBSAMPLE = 500
N_ITER      = 50


_WORKER_G = None


def _worker_init(gp: bytes) -> None:
    global _WORKER_G
    _WORKER_G = pickle.loads(gp)


def _leiden_worker(args):
    r, seed = args
    import scanpy as sc
    import anndata as ad_
    import scipy.sparse as sp
    G = _WORKER_G
    n = G.shape[0]
    a = ad_.AnnData(X=sp.csr_matrix((n, 1)))
    sc.tl.leiden(
        a, resolution=r, key_added="l",
        adjacency=G, flavor="igraph", n_iterations=2,
        directed=False, random_state=seed,
    )
    return r, a.obs["l"].astype(int).to_numpy()


def leiden_sweep(G, resolutions, seed=0):
    gp = pickle.dumps(G)
    args = [(r, seed) for r in resolutions]
    workers = min(len(args), os.cpu_count() or 1)
    out = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(gp,),
    ) as ex:
        for r, lbl in ex.map(_leiden_worker, args):
            out[r] = lbl
    return out


def _silhouette(dist_f32, labels_i32):
    try:
        from fast_auto_scrna._native import silhouette as native_sil
        return float(native_sil.silhouette_precomputed(dist_f32, labels_i32))
    except Exception:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(dist_f32, labels_i32, metric="precomputed"))


def embedding_silhouette(emb, labels_per_res, resolutions,
                         n_sub=N_SUBSAMPLE, n_iter=N_ITER, seed=0):
    """Subsample × N iterations. Euclidean on emb → Rust silhouette kernel.

    Returns per-resolution dict of {mean, sd, min, max, n_clusters}."""
    from sklearn.metrics.pairwise import euclidean_distances
    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    results = {r: [] for r in resolutions}
    ks       = {r: [] for r in resolutions}
    for _ in range(n_iter):
        idx = rng.choice(n, min(n_sub, n), replace=False)
        sub_emb = np.ascontiguousarray(emb[idx], dtype=np.float32)
        d = euclidean_distances(sub_emb).astype(np.float32)
        np.fill_diagonal(d, 0.0)
        d = np.ascontiguousarray(d)
        for r in resolutions:
            lbl = labels_per_res[r][idx].astype(np.int32)
            uniq = np.unique(lbl)
            ks[r].append(len(uniq))
            if len(uniq) < 2:
                results[r].append(0.0)
                continue
            results[r].append(_silhouette(d, np.ascontiguousarray(lbl)))
    rows = []
    for r in resolutions:
        v = np.asarray(results[r])
        rows.append({
            "resolution":  r,
            "k_med":       int(np.median(ks[r])),
            "s_mean":      float(v.mean()),
            "s_sd":        float(v.std()),
            "s_min":       float(v.min()),
            "s_max":       float(v.max()),
        })
    return pd.DataFrame(rows)


def pick_argmax_with_parsimony(df: pd.DataFrame) -> float:
    """Pick the LARGEST r whose s_mean is within 1·sd of the global max.

    Matches the user's R-workflow heuristic: stay on the max plateau,
    but prefer higher resolution (more clusters = more biological
    granularity) within the tolerance."""
    s_max = df["s_mean"].max()
    sd_at_max = df.loc[df["s_mean"].idxmax(), "s_sd"]
    eligible = df[df["s_mean"] >= s_max - sd_at_max]
    return float(eligible["resolution"].max())


def plot_curve(df, picked, out_path, title="Resolution Optimization"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    fig, ax = plt.subplots(figsize=(7, 4.5))
    r = df["resolution"].to_numpy()
    m = df["s_mean"].to_numpy()
    s = df["s_sd"].to_numpy()
    ax.plot(r, m, "k-", linewidth=1.2)
    ax.fill_between(r, m - s, m + s, color="0.75", alpha=0.5, linewidth=0)
    ax.axvline(picked, linestyle="--", color="red", linewidth=1,
               label=f"picked r={picked:.2f}")
    ax.set_xlabel("resolution")
    ax.set_ylabel("mean silhouette score (embedding)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    a = ad.read_h5ad(H5AD)
    print(f"loaded: {a.n_obs} cells, rep={REP} shape={a.obsm[REP].shape}")
    G = a.obsp["connectivities"].tocsr()
    emb = np.asarray(a.obsm[REP], dtype=np.float32)

    print(f"\nLeiden sweep ({len(RESOLUTIONS)} resolutions)...")
    t0 = time.perf_counter()
    labels_per_res = leiden_sweep(G, RESOLUTIONS, seed=0)
    t_leiden = time.perf_counter() - t0
    print(f"  leiden wall: {t_leiden:.1f}s")
    for r in RESOLUTIONS:
        k = len(np.unique(labels_per_res[r]))
        print(f"  r={r:.2f}  k={k:2d}")

    print(f"\nEmbedding silhouette ({N_ITER} iter × {N_SUBSAMPLE} cells)...")
    t0 = time.perf_counter()
    curve = embedding_silhouette(emb, labels_per_res, RESOLUTIONS,
                                 n_sub=N_SUBSAMPLE, n_iter=N_ITER)
    t_sil = time.perf_counter() - t0
    print(f"  silhouette wall: {t_sil:.1f}s")

    picked_argmax   = float(curve.loc[curve["s_mean"].idxmax(), "resolution"])
    picked_parsimon = pick_argmax_with_parsimony(curve)

    # ARI vs ground truth
    from sklearn.metrics import adjusted_rand_score
    manual = a.obs[LABEL_COL].astype(str).to_numpy()
    curve["ari_vs_manual"] = [
        adjusted_rand_score(manual, labels_per_res[r])
        for r in curve["resolution"]
    ]

    print(f"\n{'res':>6} {'k_med':>6} {'s_mean':>9} {'s_sd':>7} {'ARI':>7}")
    for _, row in curve.iterrows():
        star = ""
        if row["resolution"] == picked_argmax:   star += "  (argmax)"
        if row["resolution"] == picked_parsimon: star += "  (parsimon)"
        print(f"  {row['resolution']:.2f} {int(row['k_med']):6d} "
              f"{row['s_mean']:>9.4f} {row['s_sd']:>7.4f} "
              f"{row['ari_vs_manual']:>7.4f}{star}")

    print(f"\npicker decisions:")
    print(f"  argmax            → r={picked_argmax:.2f}  "
          f"ARI={float(curve.loc[curve['resolution']==picked_argmax, 'ari_vs_manual'].iloc[0]):.4f}")
    print(f"  plateau-parsimony → r={picked_parsimon:.2f}  "
          f"ARI={float(curve.loc[curve['resolution']==picked_parsimon, 'ari_vs_manual'].iloc[0]):.4f}")

    ari_best = float(curve["ari_vs_manual"].max())
    r_ari_best = float(curve.loc[curve["ari_vs_manual"].idxmax(), "resolution"])
    print(f"  ground truth best → r={r_ari_best:.2f}  ARI={ari_best:.4f}")

    plot_path = plot_curve(
        curve, picked_parsimon,
        "benchmarks/out/prototype_emb_silhouette_pancreas.png",
        title="Resolution Optimization (embedding silhouette, pancreas)",
    )
    print(f"\nplot: {plot_path}")


if __name__ == "__main__":
    main()

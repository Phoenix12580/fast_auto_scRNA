"""Three-way picker comparison on pancreas: conductance / silhouette / stability.

Fine resolution grid, NO k-range clip. User's directive: "don't force
(3,10) — that was for coarse-lineage first pass; first round can produce
many subclusters and be merged post-hoc by marker annotation."

Criterion for each picker: does it hit resolutions near the ground-truth
ARI peak (r=0.30 k=7 ARI=0.65 on pancreas SubCellType)?

Honest reporting — all three curves printed side by side, no spin on which
"wins". The point is to see which picker's natural choice best matches
what the data actually is, without k-range rescue.
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
from itertools import combinations


H5AD        = "benchmarks/out/smoke_pancreas.h5ad"
REP         = "X_pca"
LABEL_COL   = "SubCellType"
RESOLUTIONS = [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
               0.25, 0.30, 0.40, 0.50, 0.70, 1.00]
SEEDS       = [0, 1, 2, 3, 4]
N_SUBSAMPLE = 500
N_ITER_SILH = 50


_WORKER_G = None


def _worker_init(gp):
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
    return (r, seed, a.obs["l"].astype(int).to_numpy())


def leiden_multi_sweep(G, resolutions, seeds):
    gp = pickle.dumps(G)
    args = [(r, s) for r in resolutions for s in seeds]
    workers = min(len(args), os.cpu_count() or 1)
    out: dict[tuple, np.ndarray] = {}
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(gp,),
    ) as ex:
        for r, s, lbl in ex.map(_leiden_worker, args):
            out[(r, s)] = lbl
    return out


def _silhouette(d_f32, lbl_i32):
    try:
        from fast_auto_scrna._native import silhouette as native_sil
        return float(native_sil.silhouette_precomputed(d_f32, lbl_i32))
    except Exception:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(d_f32, lbl_i32, metric="precomputed"))


def mean_conductance(G, labels):
    from fast_auto_scrna.cluster.resolution import mean_conductance as mc
    return mc(G, labels)


def compute_curves(G, emb, labels_seed0, resolutions, results):
    """All three pickers."""
    rows = []
    from sklearn.metrics.pairwise import euclidean_distances
    rng = np.random.default_rng(0)
    n = emb.shape[0]
    # pre-sample indices for silhouette
    idx_per_iter = [
        rng.choice(n, min(N_SUBSAMPLE, n), replace=False)
        for _ in range(N_ITER_SILH)
    ]
    d_per_iter = [
        np.ascontiguousarray(
            euclidean_distances(np.ascontiguousarray(emb[idx], dtype=np.float32))
            .astype(np.float32)
        )
        for idx in idx_per_iter
    ]
    for d in d_per_iter:
        np.fill_diagonal(d, 0.0)

    for r in resolutions:
        labels_list = [results[(r, s)] for s in SEEDS]
        ks = [int(len(np.unique(l))) for l in labels_list]
        pairs = list(combinations(range(len(SEEDS)), 2))
        aris_stab = [
            __ari(labels_list[i], labels_list[j]) for i, j in pairs
        ]
        # silhouette — use seed-0 labels across subsamples
        lbl = labels_list[0]
        sil_vals = []
        for idx, d in zip(idx_per_iter, d_per_iter):
            sub_lbl = lbl[idx].astype(np.int32)
            if len(np.unique(sub_lbl)) < 2:
                sil_vals.append(0.0)
                continue
            sil_vals.append(_silhouette(d, np.ascontiguousarray(sub_lbl)))
        # conductance on full graph (seed-0)
        cond = mean_conductance(G, lbl)

        rows.append({
            "resolution":    r,
            "k_med":         int(np.median(ks)),
            "sil_mean":      float(np.mean(sil_vals)),
            "sil_sd":        float(np.std(sil_vals)),
            "conductance":   cond,
            "stab_mean_ari": float(np.mean(aris_stab)),
            "stab_sd_ari":   float(np.std(aris_stab)),
        })
    return pd.DataFrame(rows)


def __ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def main():
    a = ad.read_h5ad(H5AD)
    print(f"loaded: {a.n_obs} cells")
    G = a.obsp["connectivities"].tocsr()
    emb = np.asarray(a.obsm[REP], dtype=np.float32)
    manual = a.obs[LABEL_COL].astype(str).to_numpy()

    t0 = time.perf_counter()
    results = leiden_multi_sweep(G, RESOLUTIONS, SEEDS)
    print(f"leiden sweep ({len(RESOLUTIONS)} res × {len(SEEDS)} seeds): "
          f"{time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    df = compute_curves(G, emb, None, RESOLUTIONS, results)
    print(f"metrics wall: {time.perf_counter() - t0:.1f}s")

    df["ari_vs_manual"] = [__ari(manual, results[(r, 0)]) for r in df["resolution"]]

    # pickers, no k-clip
    pick_sil  = float(df.loc[df["sil_mean"].idxmax(),      "resolution"])
    pick_cond = float(df.loc[df["conductance"].idxmin(),   "resolution"])
    pick_stab = float(df.loc[df["stab_mean_ari"].idxmax(), "resolution"])
    pick_ari  = float(df.loc[df["ari_vs_manual"].idxmax(), "resolution"])

    print(f"\n{'r':>5} {'k':>3} {'sil':>7}±{'sd':<6} {'cond':>7}  "
          f"{'stab':>7}±{'sd':<6} {'ARI':>7}  picks")
    for _, row in df.iterrows():
        picks = []
        if row["resolution"] == pick_sil:  picks.append("S")
        if row["resolution"] == pick_cond: picks.append("C")
        if row["resolution"] == pick_stab: picks.append("T")
        if row["resolution"] == pick_ari:  picks.append("★")
        tag = "+".join(picks)
        print(f"  {row['resolution']:.2f} {int(row['k_med']):3d} "
              f"{row['sil_mean']:+7.4f}±{row['sil_sd']:.4f} "
              f"{row['conductance']:>7.4f}  "
              f"{row['stab_mean_ari']:>7.4f}±{row['stab_sd_ari']:.4f} "
              f"{row['ari_vs_manual']:>7.4f}  {tag}")

    ari_at = lambda r: float(df.loc[df['resolution']==r, 'ari_vs_manual'].iloc[0])
    print(f"\n  S = silhouette picks: r={pick_sil:.2f}  ARI={ari_at(pick_sil):.4f}")
    print(f"  C = conductance picks: r={pick_cond:.2f}  ARI={ari_at(pick_cond):.4f}")
    print(f"  T = stability picks:   r={pick_stab:.2f}  ARI={ari_at(pick_stab):.4f}")
    print(f"  ★ = ground-truth best: r={pick_ari:.2f}  ARI={ari_at(pick_ari):.4f}")


if __name__ == "__main__":
    main()

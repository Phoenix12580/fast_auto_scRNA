"""Prototype alternative resolution-selection metrics on 222k.

Candidates compared:
  (A) embedding-silhouette: Euclidean on integrated rep (50-d), subsample
  (B) mean conductance:      per-cluster boundary/volume, full graph
  (C) modularity:            Newman Q at fixed γ_eval=1.0, full graph
  (D) stability:             mean pairwise ARI across K seeds (only if asked)

Criterion: curve has a peak at a k agreeing with ct.main ARI.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import time
import numpy as np
import anndata as ad

H5AD = "benchmarks/out/smoke_222k_all_gs3.h5ad"
GRAPH = "bbknn_connectivities"
REP = "X_pca_harmony"        # bbknn_connectivities derived from this
REP_FALLBACK = "X_pca"
LABEL_COL = "ct.main"
RESOLUTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]


def embedding_silhouette(emb, labels, n_sub=1000, n_iter=100, seed=0):
    """Mean silhouette across K random subsamples, Euclidean on emb."""
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    vals = []
    for _ in range(n_iter):
        idx = rng.choice(n, min(n_sub, n), replace=False)
        sub_emb = emb[idx]
        sub_lbl = labels[idx]
        if len(np.unique(sub_lbl)) < 2:
            vals.append(0.0)
            continue
        d = euclidean_distances(sub_emb).astype(np.float32)
        np.fill_diagonal(d, 0.0)
        vals.append(float(silhouette_score(d, sub_lbl, metric="precomputed")))
    return float(np.mean(vals)), float(np.std(vals))


def mean_conductance(G, labels):
    """Mean conductance across clusters. Lower = tighter communities.

    conductance(C) = cut(C, V\\C) / min(vol(C), vol(V\\C))
    where vol(C) = sum of degrees in C.
    """
    import scipy.sparse as sp
    n = G.shape[0]
    deg = np.asarray(G.sum(axis=1)).ravel()
    total_vol = float(deg.sum())
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    conds = []
    for c in uniq:
        mask = labels == c
        vol_c = float(deg[mask].sum())
        vol_rest = total_vol - vol_c
        if vol_c == 0 or vol_rest == 0:
            continue
        G_sub = G[mask]
        cross = float(G_sub[:, ~mask].sum())
        denom = min(vol_c, vol_rest)
        conds.append(cross / denom)
    return float(np.mean(conds)) if conds else 1.0


def modularity(G, labels, gamma=1.0):
    """Newman modularity at resolution gamma. Higher = better."""
    n = G.shape[0]
    m = float(G.sum()) / 2.0   # undirected edge weight
    if m == 0:
        return 0.0
    deg = np.asarray(G.sum(axis=1)).ravel()
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    Q = 0.0
    for c in uniq:
        mask = labels == c
        # within-cluster weight (sum of A_ij over i,j in C)
        in_w = float(G[mask][:, mask].sum())
        d_c = float(deg[mask].sum())
        Q += in_w / (2.0 * m) - gamma * (d_c / (2.0 * m)) ** 2
    return float(Q)


def main():
    a = ad.read_h5ad(H5AD)
    print(f"loaded: {a.n_obs} cells")
    G = a.obsp[GRAPH].tocsr()
    rep = REP if REP in a.obsm else REP_FALLBACK
    emb = np.asarray(a.obsm[rep], dtype=np.float32)
    print(f"embedding: {rep} shape={emb.shape}")

    manual = a.obs[LABEL_COL].astype(str).to_numpy() if LABEL_COL in a.obs else None

    from sklearn.metrics import adjusted_rand_score
    results = []
    for r in RESOLUTIONS:
        key = f"leiden_bbknn_r{r:.2f}"
        if key not in a.obs.columns:
            print(f"skip r={r}: {key!r} missing")
            continue
        labels = a.obs[key].astype(int).to_numpy()
        k = len(np.unique(labels))

        t0 = time.perf_counter()
        emb_s, emb_std = embedding_silhouette(emb, labels, n_sub=1000, n_iter=20)
        t_emb = time.perf_counter() - t0

        t0 = time.perf_counter()
        cond = mean_conductance(G, labels)
        t_cond = time.perf_counter() - t0

        t0 = time.perf_counter()
        mod = modularity(G, labels, gamma=1.0)
        t_mod = time.perf_counter() - t0

        ari = adjusted_rand_score(manual, labels) if manual is not None else float("nan")

        results.append({
            "r": r, "k": k,
            "emb_silh": emb_s, "emb_silh_std": emb_std,
            "conductance": cond, "modularity": mod,
            "ari_vs_manual": ari,
            "t_emb": t_emb, "t_cond": t_cond, "t_mod": t_mod,
        })

    print()
    print(f"{'r':>6} {'k':>4} {'emb_silh±sd':>14} {'conductance':>12} "
          f"{'modularity':>12} {'ARI':>8}  times(emb/cond/mod)s")
    for row in results:
        print(
            f"  {row['r']:.2f} {row['k']:4d}  "
            f"{row['emb_silh']:+.4f}±{row['emb_silh_std']:.4f}  "
            f"{row['conductance']:>12.4f}  "
            f"{row['modularity']:>12.4f}  "
            f"{row['ari_vs_manual']:>8.4f}  "
            f"{row['t_emb']:.1f}/{row['t_cond']:.2f}/{row['t_mod']:.2f}"
        )

    # Which metric would pick the best ARI resolution?
    if manual is not None:
        best_ari = max(results, key=lambda r: r["ari_vs_manual"])
        print(f"\nground-truth best (max ARI): r={best_ari['r']:.2f} k={best_ari['k']}")
        print(f"emb_silh would pick:       r={max(results, key=lambda r: r['emb_silh'])['r']:.2f}")
        print(f"conductance (min) picks:   r={min(results, key=lambda r: r['conductance'])['r']:.2f}")
        print(f"modularity (max) picks:    r={max(results, key=lambda r: r['modularity'])['r']:.2f}")


if __name__ == "__main__":
    main()

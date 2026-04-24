"""Diagnose why graph-silhouette is monotonic in k.

Hypothesis: ``d = 1 - connectivity`` on a sparse knn graph degenerates —
most entries are 1 (non-edges), so the silhouette is dominated by noise
at the floor, and small clusters mechanically get lower a(i) because
their intra-cluster density rises (more 'real' edges per pair).

Dumps:
  - connectivity value histogram
  - fraction of edge/non-edge pairs
  - per-k: mean intra-cluster distance, mean inter-cluster distance
  - per-k: silhouette(i) histogram (not just mean)
  - ARI of each resolution's labels vs ct.main (if available)
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import anndata as ad

H5AD = "benchmarks/out/smoke_222k_all_gs3.h5ad"
GRAPH = "bbknn_connectivities"
LABEL_COL = "ct.main"


def main():
    a = ad.read_h5ad(H5AD)
    print(f"loaded: {a.n_obs} cells")
    G = a.obsp[GRAPH].tocsr()
    print(f"graph {GRAPH}: nnz={G.nnz:_}, density={G.nnz/(G.shape[0]**2):.2e}")

    # Connectivity value distribution (non-zero entries only).
    vals = G.data
    print(f"\nconnectivity value (nonzero) distribution:")
    print(f"  min={vals.min():.4f} max={vals.max():.4f} "
          f"mean={vals.mean():.4f} median={np.median(vals):.4f}")
    q = np.quantile(vals, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    print(f"  quantiles 1/10/25/50/75/90/99: "
          f"{q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f} {q[4]:.3f} "
          f"{q[5]:.3f} {q[6]:.3f}")

    # Sub-sample & study at fixed resolutions
    rng = np.random.default_rng(0)
    n_sub = 1000
    idx = rng.choice(a.n_obs, n_sub, replace=False)
    sub = G[idx][:, idx].toarray().astype(np.float32)
    print(f"\nsub-sample ({n_sub} cells):")
    print(f"  edges:      {int((sub > 0).sum() - n_sub):_} "
          f"(fraction {((sub > 0).sum() - n_sub) / (n_sub*(n_sub-1)):.4f})")
    print(f"  non-edges:  {int(((sub == 0).sum())):_}")

    dist = 1.0 - sub
    np.fill_diagonal(dist, 0.0)
    print(f"  dist stats: min={dist.min()} max={dist.max()} "
          f"fraction_of_ones={(dist == 1.0).mean():.4f}")

    # Per-resolution silhouette breakdown
    print(f"\nper-resolution breakdown (sub n={n_sub}):")
    print(f"{'res':>6} {'k':>4} {'mean_a':>10} {'mean_b':>10} "
          f"{'mean_s':>10} {'s_std':>10} {'frac_s>0':>10}")
    for r in [0.05, 0.10, 0.20, 0.30, 0.50]:
        key = f"leiden_bbknn_r{r:.2f}"
        if key not in a.obs.columns:
            print(f"  skip r={r:.2f}: obs col missing")
            continue
        lbl = a.obs[key].astype(int).to_numpy()[idx]
        uniq = np.unique(lbl)
        if len(uniq) < 2:
            continue
        from sklearn.metrics import silhouette_samples
        s_all = silhouette_samples(dist, lbl, metric="precomputed")
        # compute a/b manually
        a_vals, b_vals = [], []
        for i in range(n_sub):
            own = np.where(lbl == lbl[i])[0]
            own = own[own != i]
            if len(own) == 0:
                continue
            a_i = dist[i, own].mean()
            b_i = min(
                dist[i, np.where(lbl == c)[0]].mean()
                for c in uniq if c != lbl[i]
            )
            a_vals.append(a_i)
            b_vals.append(b_i)
        a_vals = np.array(a_vals)
        b_vals = np.array(b_vals)
        print(f"  {r:.2f}  {len(uniq):>4}  "
              f"{a_vals.mean():.4f}  {b_vals.mean():.4f}  "
              f"{s_all.mean():.4f}  {s_all.std():.4f}  "
              f"{(s_all > 0).mean():.3f}")

    # ARI vs manual annotation (if available)
    if LABEL_COL in a.obs.columns:
        from sklearn.metrics import adjusted_rand_score
        manual = a.obs[LABEL_COL].astype(str).to_numpy()
        print(f"\nARI vs {LABEL_COL!r} (n_classes={len(set(manual))}):")
        for r in [0.05, 0.10, 0.20, 0.30, 0.50]:
            key = f"leiden_bbknn_r{r:.2f}"
            if key not in a.obs.columns:
                continue
            lbl = a.obs[key].astype(int).to_numpy()
            k = len(set(lbl.tolist()))
            ari = adjusted_rand_score(manual, lbl)
            print(f"  r={r:.2f} k={k:3d}  ARI={ari:.4f}")


if __name__ == "__main__":
    main()

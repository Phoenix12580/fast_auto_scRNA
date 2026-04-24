"""Stability-based resolution picker: full-data on 222k, 3 routes.

For each integration route, run K Leiden seeds per resolution on the full
graph, then compute mean pairwise ARI. The claim we're testing:

  "stability gives a non-monotonic signal, unlike conductance which
   monotonically favors small r"

Reporting is honest. If the curve is monotonic in all routes too, the
fix doesn't work and we should say so explicitly.
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


H5AD = "benchmarks/out/smoke_222k_all_gs5.h5ad"
ROUTES = ["none", "bbknn", "harmony"]
RESOLUTIONS = [0.05, 0.1, 0.2, 0.3, 0.5]
SEEDS = [0, 1, 2, 3, 4]
LABEL_COL = "ct.main"


# ProcessPoolExecutor worker state
_WORKER_G = None


def _worker_init(graph_pickle: bytes) -> None:
    global _WORKER_G
    _WORKER_G = pickle.loads(graph_pickle)


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


def run_stability(G, resolutions, seeds, label: str):
    from sklearn.metrics import adjusted_rand_score
    gp = pickle.dumps(G)
    args = [(r, s) for r in resolutions for s in seeds]
    max_workers = min(len(args), os.cpu_count() or 1)
    print(f"\n[{label}] n={G.shape[0]}, nnz={G.nnz:_}, "
          f"jobs={len(args)}, workers={max_workers}")
    t0 = time.perf_counter()
    results: dict[tuple, np.ndarray] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(gp,),
    ) as ex:
        for r, s, lbl in ex.map(_leiden_worker, args):
            results[(r, s)] = lbl
    wall = time.perf_counter() - t0
    print(f"[{label}] sweep wall: {wall:.1f}s")

    rows = []
    for r in resolutions:
        labels_list = [results[(r, s)] for s in seeds]
        ks = [int(len(np.unique(l))) for l in labels_list]
        pairs = list(combinations(range(len(seeds)), 2))
        aris = [adjusted_rand_score(labels_list[i], labels_list[j])
                for i, j in pairs]
        rows.append({
            "resolution":  r,
            "k_med":       int(np.median(ks)),
            "k_range":     f"{min(ks)}-{max(ks)}",
            "mean_ari":    float(np.mean(aris)),
            "std_ari":     float(np.std(aris)),
            "min_ari":     float(np.min(aris)),
        })
    return pd.DataFrame(rows), results, wall


def _print_curve(label: str, df: pd.DataFrame, picked_r: float,
                 ari_manual: dict[float, float]):
    print(f"\n[{label}] stability curve:")
    print(f"{'res':>6} {'k_med':>6} {'k_range':>8} "
          f"{'mean_ari':>10} {'std_ari':>9} {'min_ari':>9} "
          f"{'ari_vs_ct':>10}")
    for _, row in df.iterrows():
        star = "  ← picked" if row["resolution"] == picked_r else ""
        r = row["resolution"]
        print(f"  {r:.2f} {row['k_med']:6d} {row['k_range']:>8} "
              f"{row['mean_ari']:>10.4f} {row['std_ari']:>9.4f} "
              f"{row['min_ari']:>9.4f} {ari_manual.get(r, float('nan')):>10.4f}{star}")


def _diagnose_monotonicity(df: pd.DataFrame) -> str:
    """Check if mean_ari is monotone (= same problem as conductance) or
    has interior peak/valley."""
    v = df["mean_ari"].to_numpy()
    diffs = np.diff(v)
    if np.all(diffs <= 1e-6):
        return "monotone non-increasing"
    if np.all(diffs >= -1e-6):
        return "monotone non-decreasing"
    # interior extremum
    argmax = int(np.argmax(v))
    if argmax == 0:
        return "peak at leftmost (r=smallest) — same shape as conductance"
    if argmax == len(v) - 1:
        return "peak at rightmost (r=largest)"
    return f"interior peak at idx={argmax} r={df['resolution'].iloc[argmax]:.2f} — USEFUL signal"


def main():
    a = ad.read_h5ad(H5AD)
    print(f"loaded: {a.n_obs} cells")
    manual = a.obs[LABEL_COL].astype(str).to_numpy()

    all_results = {}
    for route in ROUTES:
        G = a.obsp[f"{route}_connectivities"].tocsr()
        df, res_dict, wall = run_stability(G, RESOLUTIONS, SEEDS, label=route)
        # ARI vs ct.main for seed-0 labels (reference point for interpretation)
        from sklearn.metrics import adjusted_rand_score
        ari_manual = {
            r: adjusted_rand_score(manual, res_dict[(r, 0)])
            for r in RESOLUTIONS
        }
        picked_r = float(df.loc[df["mean_ari"].idxmax(), "resolution"])
        _print_curve(route, df, picked_r, ari_manual)
        shape = _diagnose_monotonicity(df)
        print(f"[{route}] curve shape: {shape}")
        print(f"[{route}] picked r={picked_r:.2f} (k_med={int(df.loc[df['resolution']==picked_r,'k_med'].iloc[0])}), "
              f"ARI vs ct.main at picked = {ari_manual[picked_r]:.4f}")
        all_results[route] = (df, picked_r, ari_manual, wall, shape)

    # final summary
    print("\n" + "=" * 72)
    print("SUMMARY — is stability a useful (non-monotonic) signal?")
    print("=" * 72)
    for route, (df, picked, ari_m, wall, shape) in all_results.items():
        print(f"  {route:8s}  wall={wall:5.1f}s  picked r={picked:.2f}  "
              f"shape: {shape}")
        print(f"            ARI@picked vs ct.main = {ari_m[picked]:.4f}")


if __name__ == "__main__":
    main()

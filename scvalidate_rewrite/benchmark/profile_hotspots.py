"""Profile each scvalidate module to identify Rust-candidate hotspots.

Runs cProfile on (1) rogue entropy + SE fit, (2) scSHC testClusters with
already-computed leiden, (3) recall's knockoff + DE loop. Prints top-20
time-consuming functions per module.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np
import pandas as pd
from scipy.io import mmread

BENCH = "F:/NMF_rewrite/scvalidate_rewrite/benchmark"


def _load():
    counts = mmread(f"{BENCH}/counts.mtx").tocsr()
    cells = open(f"{BENCH}/cells.tsv").read().strip().split("\n")
    genes = open(f"{BENCH}/genes.tsv").read().strip().split("\n")
    cl_df = pd.read_csv(f"{BENCH}/scvalidate_clusters.csv", index_col=0).loc[cells]
    return counts, cells, genes, cl_df


def _top20(pr: cProfile.Profile, label: str):
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)
    print(f"\n===== {label} (top-20 cumulative) =====")
    print(buf.getvalue())


def profile_rogue():
    # Profile the lower-level entropy + SE-fit path directly, since the
    # AnnData adapter's per-cluster sub-index chain is separately broken
    # and isn't the kernel we'd Rust-rewrite anyway.
    from scvalidate.rogue_py.core import entropy_table, entropy_fit

    counts, _, _, cl_df = _load()
    X_full = counts.toarray()  # genes × cells
    clusters = cl_df["leiden"].astype(str).to_numpy()

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for cid in np.unique(clusters):
        mask = clusters == cid
        if mask.sum() < 10:
            continue
        Xc = X_full[:, mask]
        ent = entropy_table(Xc)
        entropy_fit(ent, span=0.5)
    pr.disable()
    print(f"rogue entropy + SE-fit loop: {time.perf_counter()-t0:.1f}s")
    _top20(pr, "ROGUE entropy")


def profile_scshc():
    from scvalidate.scshc_py import test_clusters

    counts, _, _, cl_df = _load()
    data = np.asarray(counts.todense())
    leiden = cl_df["leiden"].astype(str).to_numpy()

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    test_clusters(data, leiden, alpha=0.05, num_features=2000, num_pcs=30)
    pr.disable()
    print(f"scshc testClusters: {time.perf_counter()-t0:.1f}s")
    _top20(pr, "sc-SHC")


def profile_recall():
    import scanpy as sc
    from anndata import AnnData
    from scvalidate.recall_py import find_clusters_recall

    counts, cells, genes, _ = _load()
    X = counts.T.tocsr()
    adata = AnnData(X=X.astype(np.float64))
    adata.obs_names = cells
    adata.var_names = genes

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    find_clusters_recall(adata, max_iterations=1)
    pr.disable()
    print(f"recall find_clusters_recall (1 iter): {time.perf_counter()-t0:.1f}s")
    _top20(pr, "recall")


if __name__ == "__main__":
    print("### Profiling scSHC ###")
    profile_scshc()
    print("\n### Profiling ROGUE ###")
    profile_rogue()
    print("\n### Profiling recall ###")
    profile_recall()

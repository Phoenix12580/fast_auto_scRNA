"""Microbench: single Leiden vs sequential sweep vs parallel sweep.

Before committing to a Rust Leiden port, quantify the ceiling. Measures:
  1. single sc.tl.leiden at r=0.2 on the 222k bbknn graph
  2. sequential 5-resolution sweep [0.05, 0.1, 0.2, 0.3, 0.5]
  3. same sweep via ProcessPoolExecutor (spawn, Windows)

If Python MP buys us ≥ 3x, Rust Leiden is probably not worth the risk to
result-parity. If MP is crippled by IPC, Rust Leiden becomes attractive.
"""
from __future__ import annotations

import time
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


H5AD = "benchmarks/out/smoke_222k_all_gs3.h5ad"
GRAPH_KEY = "bbknn_connectivities"
RESOLUTIONS = [0.05, 0.1, 0.2, 0.3, 0.5]


def _load_graph():
    import anndata as ad
    a = ad.read_h5ad(H5AD)
    G = a.obsp[GRAPH_KEY].tocsr()
    print(f"  graph: {G.shape[0]} x {G.shape[1]}, nnz={G.nnz:_}, "
          f"data.nbytes={G.data.nbytes/1e6:.1f} MB")
    return a, G


def _leiden_one(args):
    graph_pickle, resolution, seed = args
    import scanpy as sc
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    G = pickle.loads(graph_pickle)
    n = G.shape[0]
    a = ad.AnnData(X=sp.csr_matrix((n, 1)))
    sc.tl.leiden(
        a, resolution=resolution, key_added="leiden",
        adjacency=G, flavor="igraph", n_iterations=2,
        directed=False, random_state=seed,
    )
    return resolution, a.obs["leiden"].astype(int).to_numpy()


def _bench_single(a, G):
    import scanpy as sc
    t0 = time.perf_counter()
    sc.tl.leiden(
        a, resolution=0.2, key_added="_bench_r02",
        adjacency=G, flavor="igraph", n_iterations=2,
        directed=False, random_state=0,
    )
    dt = time.perf_counter() - t0
    k = a.obs["_bench_r02"].nunique()
    print(f"[1] single Leiden @ r=0.2 : {dt:6.2f}s  (k={k})")
    return dt


def _bench_sequential(a, G):
    import scanpy as sc
    t0 = time.perf_counter()
    for r in RESOLUTIONS:
        sc.tl.leiden(
            a, resolution=r, key_added=f"_bench_seq_r{r}",
            adjacency=G, flavor="igraph", n_iterations=2,
            directed=False, random_state=0,
        )
    dt = time.perf_counter() - t0
    print(f"[2] sequential x{len(RESOLUTIONS)}  : {dt:6.2f}s  "
          f"(avg {dt/len(RESOLUTIONS):.2f}s/res)")
    return dt


def _bench_parallel(G, max_workers=5):
    graph_pickle = pickle.dumps(G)
    print(f"  pickled graph : {len(graph_pickle)/1e6:.1f} MB")
    args = [(graph_pickle, r, 0) for r in RESOLUTIONS]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_leiden_one, args))
    dt = time.perf_counter() - t0
    ks = [len(set(lbl.tolist())) for _, lbl in results]
    print(f"[3] parallel x{len(RESOLUTIONS)} (w={max_workers}): "
          f"{dt:6.2f}s  (ks={ks})")
    return dt


def main():
    print(f"loading {H5AD}")
    a, G = _load_graph()

    dt_single   = _bench_single(a, G)
    dt_seq      = _bench_sequential(a, G)
    dt_par      = _bench_parallel(G, max_workers=5)

    print()
    print("summary")
    print(f"  sequential / single     = {dt_seq/dt_single:5.2f}x  (ideal {len(RESOLUTIONS)}x)")
    print(f"  parallel   / sequential = {dt_par/dt_seq:5.2f}x       (lower = better)")
    print(f"  parallel speedup        = {dt_seq/dt_par:5.2f}x over sequential")


if __name__ == "__main__":
    main()

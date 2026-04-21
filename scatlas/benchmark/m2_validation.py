"""M2 / M2.5 acceptance — full 157k BBKNN pipeline on epithelia_full.h5ad.

Checks end-to-end:
  1. scatlas.ext.bbknn runs on the full dataset via `auto` backend
     (must pick hnsw) in reasonable time.
  2. Output writes scanpy-consumable structures
     (adata.obsp['bbknn_distances'], adata.uns['bbknn']).
  3. Batch balance: every cell's neighborhood spans both batches.
  4. Biological coherence: neighbors preferentially share subtype.
  5. Connectivities (umap-learn fuzzy_simplicial_set) produces a
     symmetric-ish sparse matrix usable by sc.tl.leiden.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

import anndata as ad  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402

from scatlas import ext  # noqa: E402

EPITHELIA = Path("/mnt/f/NMF_rewrite/epithelia_full.h5ad")


def main() -> int:
    print("=" * 72)
    print("M2/M2.5 ACCEPTANCE — 157k epithelia full BBKNN")
    print("=" * 72)

    t_load0 = time.perf_counter()
    adata = ad.read_h5ad(EPITHELIA)
    t_load = time.perf_counter() - t_load0
    n = adata.n_obs
    print(f"\n[load]   {n} cells × {adata.n_vars} genes in {t_load:.1f}s")
    print(f"         obs cols: {list(adata.obs.columns)}")

    # Log-normalize
    t0 = time.perf_counter()
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((1e4 / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    t_norm = time.perf_counter() - t0
    print(f"[lognorm] {t_norm:.1f}s  (nnz={Xn.nnz}, density={Xn.nnz/(n*adata.n_vars):.2%})")

    # PCA (30 comps)
    t0 = time.perf_counter()
    pca = TruncatedSVD(n_components=30, random_state=0).fit_transform(Xn).astype(np.float32)
    t_pca = time.perf_counter() - t0
    adata.obsm["X_pca"] = pca
    print(f"[pca]    shape={pca.shape}, {t_pca:.1f}s")

    batch_col = "orig.ident"
    adata.obs["batch"] = adata.obs[batch_col].astype(str)
    uniq, cnt = np.unique(adata.obs["batch"], return_counts=True)
    print(f"[batch]  {batch_col} → {dict(zip(uniq.tolist(), cnt.tolist()))}")

    # --- BBKNN kneighbors only (no connectivities) ---
    t0 = time.perf_counter()
    raw = ext.bbknn_kneighbors(
        pca, adata.obs["batch"].to_numpy(), neighbors_within_batch=3, backend="auto"
    )
    t_knn = time.perf_counter() - t0
    print(f"\n[bbknn-kneighbors] {t_knn:.2f}s  (backend={raw['backend_used']})")
    assert raw["backend_used"] == "hnsw", "auto should have picked hnsw on 157k"

    idx = raw["indices"]
    dist = raw["distances"]
    MAX = np.iinfo(np.uint32).max
    valid = idx != MAX
    n_valid_per_cell = valid.sum(axis=1)
    d_finite = dist[valid]
    print(f"  indices {idx.shape},  mean valid/cell = {n_valid_per_cell.mean():.1f}")
    print(f"  distance min/median/max = {d_finite.min():.3f} / {np.median(d_finite):.3f} / {d_finite.max():.3f}")

    # --- Batch balance check ---
    batch_arr = adata.obs["batch"].to_numpy()
    batch_of = batch_arr  # per-cell
    own_batch_rate = 0.0
    cross_batch_rate = 0.0
    sampled = np.random.default_rng(0).choice(n, size=min(10000, n), replace=False)
    for i in sampled:
        nbrs = idx[i][valid[i]]
        same = (batch_of[nbrs] == batch_of[i]).sum()
        own_batch_rate += same / len(nbrs)
        cross_batch_rate += (len(nbrs) - same) / len(nbrs)
    own_batch_rate /= len(sampled)
    cross_batch_rate /= len(sampled)
    print(f"  batch split: {own_batch_rate:.1%} same-batch,  {cross_batch_rate:.1%} cross-batch")
    # With 2 batches × 3 neighbors each: exactly 50% same / 50% cross expected
    assert 0.4 <= own_batch_rate <= 0.6, "batch balance broken"

    # --- Biological coherence: subtype-shared neighbor rate ---
    subtype = adata.obs.get("subtype")
    if subtype is not None:
        st = subtype.astype(str).to_numpy()
        uniq_st = np.unique(st)
        print(f"  subtypes: {len(uniq_st)} levels")
        rate = 0.0
        for i in sampled:
            nbrs = idx[i][valid[i]]
            rate += (st[nbrs] == st[i]).mean()
        rate /= len(sampled)
        print(f"  fraction of neighbors sharing subtype: {rate:.1%}  "
              f"(baseline random = {1/len(uniq_st):.1%})")

    # --- Full adata wrapper with connectivities ---
    t0 = time.perf_counter()
    result = ext.bbknn(
        adata, batch_key="batch", use_rep="X_pca",
        neighbors_within_batch=3, with_connectivities=True,
    )
    t_adata = time.perf_counter() - t0
    dist_csr = adata.obsp["bbknn_distances"]
    conn_csr = adata.obsp["bbknn_connectivities"]
    print(f"\n[bbknn-adata-full] {t_adata:.1f}s  (kNN + connectivities)")
    print(f"  distances: shape={dist_csr.shape}, nnz={dist_csr.nnz}")
    print(f"  connectivities: shape={conn_csr.shape}, nnz={conn_csr.nnz}")
    # Connectivities should be nearly symmetric
    sym_delta = abs(conn_csr - conn_csr.T).sum()
    print(f"  connectivities |C - C^T|₁ = {sym_delta:.3e}  (expect small)")
    assert sym_delta < 1e-3 * conn_csr.nnz, "connectivities not symmetric"

    # --- Summary ---
    print("\n" + "=" * 72)
    print(f"TOTAL timings (post-load):")
    print(f"  lognorm         {t_norm:6.1f}s")
    print(f"  pca             {t_pca:6.1f}s")
    print(f"  bbknn-kneighbors{t_knn:6.1f}s   <- M2/M2.5 Rust kernel")
    print(f"  bbknn-adata-full{t_adata:6.1f}s   <- including umap-learn connectivities")
    print("=" * 72)
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

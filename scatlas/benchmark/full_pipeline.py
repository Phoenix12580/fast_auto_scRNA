"""Full scatlas pipeline benchmark on 157k epithelia.

Exercises every Rust kernel scatlas currently ships:
  * pp.pca — randomized truncated SVD on sparse CSR (M5.1)
  * ext.bbknn — batch-balanced kNN with HNSW backend (M2/M2.5)
  * ext.harmony — Harmony 2.0 integration, Fixed λ (M4.1-4.3)
  * metrics.scib_score — iLISI + cLISI + graph_connectivity + kBET (M3/M3.3)

Compares against a "scanpy/scib baseline" stack (sklearn TruncatedSVD +
scanpy.pp.neighbors + harmonypy + scib-metrics). Baseline is skipped if
the packages aren't installed.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

import anndata as ad  # noqa: E402

import scatlas  # noqa: E402
from scatlas import ext, metrics, pp, tl  # noqa: E402

EPITHELIA = Path("/mnt/f/NMF_rewrite/epithelia_full.h5ad")


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def main() -> int:
    banner("scatlas full pipeline bench — 157k epithelia")
    print(f"scatlas version: {scatlas.__version__}")

    t_all0 = time.perf_counter()

    t0 = time.perf_counter()
    adata = ad.read_h5ad(EPITHELIA)
    t_load = time.perf_counter() - t0
    print(f"\n[1/6 load]    {adata.n_obs} × {adata.n_vars}, {t_load:.1f}s")

    t0 = time.perf_counter()
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((1e4 / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    adata.X = Xn.tocsr()
    t_norm = time.perf_counter() - t0
    print(f"[2/6 lognorm] {t_norm:.1f}s  nnz={adata.X.nnz}")

    # --- M5.1 PCA with Gavish-Donoho auto n_comps ---
    t0 = time.perf_counter()
    pca_out = pp.pca(adata, n_comps="auto", random_state=0)
    t_pca = time.perf_counter() - t0
    auto_info = pca_out["auto"]
    print(f"[3/6 pca]     {t_pca:.1f}s  obsm['X_pca'] {adata.obsm['X_pca'].shape}")
    print(f"             auto: GD={auto_info['n_comps_gavish_donoho']}, "
          f"elbow={auto_info['n_comps_elbow']}, "
          f"final={auto_info['suggested_n_comps']}")

    # --- batch column ---
    adata.obs["batch"] = adata.obs["orig.ident"].astype(str)
    uniq, cnt = np.unique(adata.obs["batch"], return_counts=True)
    print(f"             batches: {dict(zip(uniq.tolist(), cnt.tolist()))}")

    # --- M2.5 BBKNN (with connectivities since UMAP needs them) ---
    t0 = time.perf_counter()
    raw_knn = ext.bbknn_kneighbors(
        adata.obsm["X_pca"],
        adata.obs["batch"].to_numpy(),
        neighbors_within_batch=3,
        backend="auto",
    )
    ext.bbknn(
        adata, batch_key="batch", use_rep="X_pca",
        neighbors_within_batch=3, with_connectivities=True,
    )
    t_bbknn = time.perf_counter() - t0
    print(f"[4/7 bbknn]   {t_bbknn:.1f}s  backend={raw_knn['backend_used']}"
          f"  conn_nnz={adata.obsp['bbknn_connectivities'].nnz}")

    # --- M3/M3.3 scib metrics ---
    batch_arr = adata.obs["batch"].to_numpy()
    subtype_arr = (
        adata.obs["subtype"].astype(str).to_numpy()
        if "subtype" in adata.obs.columns
        else batch_arr
    )
    # raw_knn['indices'] is uint32 with u32::MAX sentinels for padded
    # slots (when a batch has fewer than k cells). Replace sentinels
    # with self-index (distance 0) so LISI/kBET see a clean (N, k).
    indices_u32 = raw_knn["indices"]
    sentinel_mask = indices_u32 == np.iinfo(np.uint32).max
    knn_idx = indices_u32.astype(np.int32)
    knn_dist = raw_knn["distances"].astype(np.float32).copy()
    if sentinel_mask.any():
        row_idx = np.broadcast_to(
            np.arange(knn_idx.shape[0], dtype=np.int32)[:, None], knn_idx.shape
        )
        knn_idx = np.where(sentinel_mask, row_idx, knn_idx)
        knn_dist[sentinel_mask] = 0.0

    t0 = time.perf_counter()
    scib = metrics.scib_score(
        knn_idx, knn_dist,
        batch_labels=batch_arr,
        label_labels=subtype_arr,
    )
    t_scib = time.perf_counter() - t0
    print(f"[5/7 metrics] scib_score {t_scib:.2f}s")
    for k_, v_ in scib.items():
        if isinstance(v_, float):
            print(f"             {k_:20s} = {v_:.3f}")

    # --- M4.3 Harmony ---
    t0 = time.perf_counter()
    ext.harmony(
        adata, batch_key="batch", use_rep="X_pca",
        n_clusters=100, seed=0,
    )
    t_harmony = time.perf_counter() - t0
    print(f"[6/7 harmony] {t_harmony:.1f}s  obsm['X_pca_harmony'] {adata.obsm['X_pca_harmony'].shape}")

    # --- M7.1 UMAP on BBKNN connectivities ---
    t0 = time.perf_counter()
    tl.umap(adata, init="pca", random_state=0, n_epochs=200)
    t_umap = time.perf_counter() - t0
    print(f"[7/7 umap]    {t_umap:.1f}s  obsm['X_umap'] {adata.obsm['X_umap'].shape}")

    t_all = time.perf_counter() - t_all0
    banner("TIMINGS")
    print(f"  load       {t_load:6.1f}s  (IO)")
    print(f"  lognorm    {t_norm:6.1f}s  (scipy)")
    print(f"  pca        {t_pca:6.1f}s  ← scatlas M5.1/M5.2 (auto)")
    print(f"  bbknn      {t_bbknn:6.1f}s  ← scatlas M2.5")
    print(f"  metrics    {t_scib:6.1f}s  ← scatlas M3.3")
    print(f"  harmony    {t_harmony:6.1f}s  ← scatlas M4.3")
    print(f"  umap       {t_umap:6.1f}s  ← scatlas M7.1")
    print(f"  ---------------")
    print(f"  TOTAL      {t_all:6.1f}s  ({t_all / 60:.1f} min)")
    t_rust = t_pca + t_bbknn + t_scib + t_harmony + t_umap
    print(f"  (rust kernels only: {t_rust:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

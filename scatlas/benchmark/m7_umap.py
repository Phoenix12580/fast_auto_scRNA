"""M7.1 UMAP benchmark on 157k epithelia.

Compares scatlas.tl.umap (Rust SGD + Hogwild rayon) vs
scanpy.tl.umap (umap-learn numba single-thread).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

import anndata as ad  # noqa: E402

from scatlas import ext, pp, tl  # noqa: E402

EPITHELIA = Path("/mnt/f/NMF_rewrite/epithelia_full.h5ad")


def trustworthiness_knn(X_high: np.ndarray, X_low: np.ndarray, k: int = 15,
                        n_sub: int = 5000) -> float:
    """Fraction of high-dim k-NN preserved as low-dim k-NN (sampled).
    This is the biologically-meaningful UMAP quality metric — high-dim
    local structure kept intact in 2D."""
    from sklearn.neighbors import NearestNeighbors
    n = X_high.shape[0]
    n_sub = min(n_sub, n)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, n_sub, replace=False)
    nn_h = NearestNeighbors(n_neighbors=k + 1).fit(X_high).kneighbors(
        X_high[idx], return_distance=False
    )[:, 1:]
    nn_l = NearestNeighbors(n_neighbors=k + 1).fit(X_low).kneighbors(
        X_low[idx], return_distance=False
    )[:, 1:]
    preserved = 0.0
    for i in range(n_sub):
        preserved += len(set(nn_h[i]) & set(nn_l[i])) / k
    return preserved / n_sub


def main() -> int:
    print("=" * 72)
    print("M7.1 UMAP bench — 157k epithelia × 16337 genes")
    print("=" * 72)

    t0 = time.perf_counter()
    adata = ad.read_h5ad(EPITHELIA)
    print(f"[load]    {adata.n_obs} × {adata.n_vars}, {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((1e4 / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    adata.X = Xn.tocsr()
    print(f"[lognorm] {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    pp.pca(adata, n_comps="auto", random_state=0)
    print(f"[pca]     {time.perf_counter() - t0:.1f}s  ({adata.obsm['X_pca'].shape})")

    adata.obs["batch"] = adata.obs["orig.ident"].astype(str)
    t0 = time.perf_counter()
    ext.bbknn(
        adata, batch_key="batch", use_rep="X_pca",
        neighbors_within_batch=3, with_connectivities=True,
    )
    print(f"[bbknn]   {time.perf_counter() - t0:.1f}s  "
          f"conn nnz={adata.obsp['bbknn_connectivities'].nnz}")

    # --- scatlas UMAP ---
    print("\n--- scatlas.tl.umap ---")
    t0 = time.perf_counter()
    emb_sc = tl.umap(adata, init="pca", random_state=0, n_epochs=200)
    t_sc = time.perf_counter() - t0
    params = adata.uns["umap"]
    print(f"  {t_sc:.1f}s, n_epochs={params['params']['n_epochs']}, "
          f"a={params['a']:.3f}, b={params['b']:.3f}")
    print(f"  emb range: x=[{emb_sc[:,0].min():.2f}, {emb_sc[:,0].max():.2f}], "
          f"y=[{emb_sc[:,1].min():.2f}, {emb_sc[:,1].max():.2f}]")

    # --- umap-learn's layout optimizer on the same BBKNN graph ---
    # Use umap.umap_.simplicial_set_embedding so both implementations see
    # the identical connectivity matrix (apples-to-apples SGD comparison).
    try:
        from umap.umap_ import simplicial_set_embedding, find_ab_params
        a, b = find_ab_params(spread=1.0, min_dist=0.5)
        conn = adata.obsp["bbknn_connectivities"].tocoo()
        n = adata.n_obs
        # Same PCA-derived init as scatlas for fair comparison.
        pca_init = np.ascontiguousarray(
            adata.obsm["X_pca"][:, :2], dtype=np.float32
        )
        std = pca_init.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        pca_init = (pca_init / std * 1e-4).astype(np.float32)
        pca_init = pca_init + (1e-4 * np.random.default_rng(0).standard_normal(
            pca_init.shape
        )).astype(np.float32)

        print("\n--- umap-learn simplicial_set_embedding (numba single-thread) ---")
        t0 = time.perf_counter()
        emb_sk, _ = simplicial_set_embedding(
            data=None,
            graph=adata.obsp["bbknn_connectivities"].tocsr(),
            n_components=2,
            initial_alpha=1.0,
            a=a, b=b,
            gamma=1.0,
            negative_sample_rate=5,
            n_epochs=200,
            init=pca_init,
            random_state=np.random.RandomState(0),
            metric="euclidean",
            metric_kwds={},
            densmap=False,
            densmap_kwds={},
            output_dens=False,
        )
        t_sk = time.perf_counter() - t0
        print(f"  {t_sk:.1f}s")

        # Quality: fraction of PCA-space 15-NN that survive to 2D.
        # Both impls should give ≈ 0.35-0.5 on real scRNA data.
        tw_sc = trustworthiness_knn(adata.obsm["X_pca"], emb_sc, k=15)
        tw_sk = trustworthiness_knn(adata.obsm["X_pca"], emb_sk, k=15)
        print(f"\n[quality] high-dim 15-NN preservation (higher = better):")
        print(f"  scatlas     = {tw_sc:.3f}")
        print(f"  umap-learn  = {tw_sk:.3f}")

        print("\n" + "=" * 72)
        print(f"umap-learn (single-thread): {t_sk:6.1f}s  (trustworthiness {tw_sk:.3f})")
        print(f"scatlas.tl.umap (rayon)   : {t_sc:6.1f}s  (trustworthiness {tw_sc:.3f})")
        print(f"  speedup                 : {t_sk / t_sc:.2f}×")
        print("=" * 72)
    except ImportError:
        print("\numap-learn not available — skipping parity comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

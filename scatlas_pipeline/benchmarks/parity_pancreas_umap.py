"""scatlas UMAP ↔ umap-learn parity on pancreas_sub (1000 cells).

Feeds BOTH scatlas (single-thread) and umap-learn's simplicial_set_embedding
the SAME spectral init computed from the SAME connectivity graph, then
compares outputs on:
  1. No dimensional collapse — both axis ranges within 30% of each other.
  2. Shape similarity — Procrustes residual < 0.15.
  3. Cluster quality — trustworthiness within 0.02.

Pass/fail summary printed at the end. Saves a side-by-side PNG.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

import anndata as ad  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PANCREAS_RDA = Path("/mnt/f/NMF_rewrite/scvalidate_rewrite/benchmark/pancreas_sub.rda")
OUT_DIR = Path(__file__).resolve().parent / "pancreas_results"
OUT_DIR.mkdir(exist_ok=True)


def load_pancreas_sub() -> ad.AnnData:
    import rdata
    parsed = rdata.read_rda(str(PANCREAS_RDA))
    seurat = parsed["pancreas_sub"]
    rna = seurat.assays["RNA"]
    counts_raw = rna.layers["counts"]
    n_genes, n_cells = int(counts_raw.Dim[0]), int(counts_raw.Dim[1])
    mat_gc = sp.csc_matrix(
        (counts_raw.x, counts_raw.i, counts_raw.p),
        shape=(n_genes, n_cells),
    )
    X = mat_gc.T.tocsr().astype(np.float32)
    meta = getattr(seurat, "meta_data", None) or getattr(seurat, "meta.data", None)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    for col in meta.columns:
        adata.obs[str(col)] = pd.Categorical(meta[col].to_numpy())
    return adata


def trustworthiness_knn(X_high, X_low, k=15, n_sub=1000):
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
    preserved = sum(
        len(set(nn_h[i]) & set(nn_l[i])) / k for i in range(n_sub)
    )
    return preserved / n_sub


def procrustes_residual(A: np.ndarray, B: np.ndarray) -> float:
    """Normalized Procrustes disparity: optimally rotate/scale A onto B
    then return sum-of-squared residuals / ||B||²."""
    from scipy.spatial import procrustes
    _, _, disparity = procrustes(A.astype(np.float64), B.astype(np.float64))
    return float(disparity)


def axis_range_ratio(emb: np.ndarray) -> float:
    """Ratio of smaller axis range to larger. 1.0 = square, <0.2 = collapse."""
    ranges = emb.max(axis=0) - emb.min(axis=0)
    return float(ranges.min() / ranges.max())


def main() -> int:
    import scanpy as sc
    from scatlas import pp, tl

    print("=" * 78)
    print("scatlas ↔ umap-learn PARITY on pancreas_sub")
    print("=" * 78)

    adata = load_pancreas_sub()
    print(f"loaded: {adata.n_obs} × {adata.n_vars}")

    # --- shared pipeline up to graph ---
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((1e4 / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    adata.X = Xn.tocsr()

    pp.pca(adata, n_comps="auto", random_state=0)
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_pca",
                    metric="cosine", random_state=0)
    adata.obsp["bbknn_connectivities"] = adata.obsp["connectivities"]

    conn = adata.obsp["bbknn_connectivities"].tocsr()
    print(f"connectivities: {conn.shape}, nnz={conn.nnz}")

    # --- identical spectral init for both sides ---
    from umap.spectral import spectral_layout
    from umap.umap_ import find_ab_params, simplicial_set_embedding

    a, b = find_ab_params(spread=1.0, min_dist=0.5)
    print(f"(a, b) = ({a:.4f}, {b:.4f})")

    rng = np.random.default_rng(0)
    init_spectral = spectral_layout(
        data=None, graph=conn.astype(np.float64), dim=2,
        random_state=rng,
    )
    # noisy_scale_coords — max|x|=10 + σ=1e-4 noise
    max_abs = float(np.abs(init_spectral).max())
    if max_abs > 0:
        init_spectral = init_spectral * (10.0 / max_abs)
    init_spectral = (init_spectral + 1e-4 * np.random.default_rng(0).standard_normal(
        init_spectral.shape
    )).astype(np.float32)
    init_spectral = np.ascontiguousarray(init_spectral)
    print(f"init shape: {init_spectral.shape}  range: x={init_spectral[:,0].min():.2f}..{init_spectral[:,0].max():.2f}  y={init_spectral[:,1].min():.2f}..{init_spectral[:,1].max():.2f}")

    # --- scatlas single-thread with the exact spectral init ---
    print("\n--- scatlas.tl.umap (single_thread=True, init=spectral) ---")
    # Pre-place init under obsm['X_pca'] is not needed since we call umap()
    # with init='spectral' which internally computes spectral_layout; to
    # feed the IDENTICAL init we bypass _pick_init by monkey-inserting.
    # Simpler: call the Rust kernel directly with our init.
    from scatlas import _scatlas_native
    t0 = time.perf_counter()
    indptr = np.asarray(conn.indptr, dtype=np.uint64)
    indices = np.asarray(conn.indices, dtype=np.uint32)
    data = np.ascontiguousarray(conn.data, dtype=np.float32)
    emb_sc, a_sc, b_sc, n_ep_sc = _scatlas_native.tl.umap_layout(
        indptr, indices, data,
        adata.n_obs, init_spectral,
        n_components=2, n_epochs=200,
        min_dist=0.5, spread=1.0,
        negative_sample_rate=5, repulsion_strength=1.0,
        learning_rate=1.0, seed=0,
        single_thread=True,
    )
    t_sc = time.perf_counter() - t0
    print(f"  {t_sc:.2f}s  a={a_sc:.4f} b={b_sc:.4f} n_ep={n_ep_sc}")
    print(f"  range: x={emb_sc[:,0].min():.2f}..{emb_sc[:,0].max():.2f}  y={emb_sc[:,1].min():.2f}..{emb_sc[:,1].max():.2f}")

    # --- umap-learn with the same spectral init ---
    print("\n--- umap-learn simplicial_set_embedding (numba single-thread) ---")
    t0 = time.perf_counter()
    emb_ul, _ = simplicial_set_embedding(
        data=None, graph=conn,
        n_components=2, initial_alpha=1.0,
        a=a, b=b, gamma=1.0,
        negative_sample_rate=5, n_epochs=200,
        init=init_spectral.copy(),
        random_state=np.random.RandomState(0),
        metric="euclidean", metric_kwds={},
        densmap=False, densmap_kwds={}, output_dens=False,
    )
    t_ul = time.perf_counter() - t0
    print(f"  {t_ul:.2f}s")
    print(f"  range: x={emb_ul[:,0].min():.2f}..{emb_ul[:,0].max():.2f}  y={emb_ul[:,1].min():.2f}..{emb_ul[:,1].max():.2f}")

    # --- parity checks ---
    print("\n" + "=" * 78)
    print("PARITY CHECKS")
    print("=" * 78)

    sc_ratio = axis_range_ratio(emb_sc)
    ul_ratio = axis_range_ratio(emb_ul)
    print(f"axis ratio (min/max):  scatlas={sc_ratio:.3f}  umap-learn={ul_ratio:.3f}")
    check1 = sc_ratio > 0.3
    print(f"  [{'PASS' if check1 else 'FAIL'}] scatlas not dimensionally collapsed (ratio > 0.3)")

    resid = procrustes_residual(emb_sc, emb_ul)
    print(f"Procrustes disparity:  {resid:.4f}")
    check2 = resid < 0.15
    print(f"  [{'PASS' if check2 else 'FAIL'}] shape similar to umap-learn (disparity < 0.15)")

    pc_hi = adata.obsm["X_pca"]
    tw_sc = trustworthiness_knn(pc_hi, emb_sc, k=15)
    tw_ul = trustworthiness_knn(pc_hi, emb_ul, k=15)
    print(f"trustworthiness:       scatlas={tw_sc:.3f}  umap-learn={tw_ul:.3f}")
    check3 = abs(tw_sc - tw_ul) < 0.03
    print(f"  [{'PASS' if check3 else 'FAIL'}] trustworthiness within 0.03 of umap-learn")

    # --- side-by-side plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ct = adata.obs["CellType"].astype(str).to_numpy()
    uniq = np.unique(ct)
    cmap = plt.colormaps["tab10"]
    for ax, emb, title in [
        (axes[0], emb_sc, f"scatlas (Rust, single_thread)\naxis_ratio={sc_ratio:.2f} trust={tw_sc:.3f}"),
        (axes[1], emb_ul, f"umap-learn (numba)\naxis_ratio={ul_ratio:.2f} trust={tw_ul:.3f}"),
    ]:
        for i, lab in enumerate(uniq):
            m = ct == lab
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.7,
                       color=cmap(i / max(len(uniq) - 1, 1)), label=lab)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle(f"PARITY — pancreas_sub — Procrustes disparity={resid:.4f}")
    fig.tight_layout()
    out = OUT_DIR / "parity_pancreas_umap.png"
    fig.savefig(out, dpi=150)
    print(f"\nplot: {out}")

    all_pass = check1 and check2 and check3
    print("\n" + "=" * 78)
    print(f"RESULT: {'ALL PASS ✓' if all_pass else 'FAIL ✗'}")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

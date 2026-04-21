"""pancreas_sub.rda — SCOP single-sample demo (standard_scop).

Dataset:
  1000 cells × 15998 genes, 1 batch (SeuratProject), 5 CellType,
  8 SubCellType (pancreatic development lineage:
  Ductal / Endocrine(Alpha/Beta/Delta/Epsilon) / Ngn3-high-EP /
  Ngn3-low-EP / Pre-endocrine).

Head-to-head:
  1. SCOP-aligned = scanpy with SCOP defaults
     (LogNormalize, vst HVG 2000, scale, 50 PCs, kNN k=20 cosine,
      UMAP n_neighbors=30 min_dist=0.3 cosine, Louvain res=0.6)
     + pure-Python scvalidate recall
  2. scatlas-optimized = scatlas.pp.pca (GD auto) + scatlas standard kNN
     from PCA (no BBKNN since 1 batch) + scatlas.tl.umap + Leiden
     + Rust scvalidate recall
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

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PANCREAS_RDA = Path("/mnt/f/NMF_rewrite/scvalidate_rewrite/benchmark/pancreas_sub.rda")
OUT_DIR = Path(__file__).resolve().parent / "pancreas_results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def load_pancreas_sub() -> ad.AnnData:
    """Seurat rds → AnnData (cells × genes)."""
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


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def run_scop_aligned(adata: ad.AnnData) -> dict:
    """scanpy pipeline with SCOP standard_scop defaults + pure-Py recall."""
    import scanpy as sc

    t_total = time.perf_counter()
    phases: dict[str, float] = {}

    t = time.perf_counter()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    phases["lognorm"] = time.perf_counter() - t

    t = time.perf_counter()
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    ad_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(ad_hvg, max_value=10)
    sc.tl.pca(ad_hvg, n_comps=50, random_state=0)
    adata.obsm["X_pca"] = ad_hvg.obsm["X_pca"]
    phases["pca"] = time.perf_counter() - t

    # kNN graph — SCOP uses k=20 cosine on PCA
    t = time.perf_counter()
    sc.pp.neighbors(
        adata, n_neighbors=20, n_pcs=50,
        metric="cosine", random_state=0,
    )
    phases["neighbors"] = time.perf_counter() - t

    # UMAP — SCOP: n_neighbors=30 already inside sc.pp.neighbors call above;
    # but sc.tl.umap uses its own kNN recomputation iff key missing. Just
    # pass min_dist/spread here.
    t = time.perf_counter()
    sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=0)
    phases["umap"] = time.perf_counter() - t

    # Louvain res=0.6
    t = time.perf_counter()
    sc.tl.louvain(
        adata, resolution=0.6, random_state=0,
        flavor="igraph", directed=False,
    )
    adata.obs["cluster"] = adata.obs["louvain"]
    phases["cluster"] = time.perf_counter() - t
    print(f"  [scop] louvain → {adata.obs['cluster'].nunique()} clusters")

    # recall with pure-Python wilcoxon
    t = time.perf_counter()
    try:
        from scvalidate.recall_py import core as _rcore, knockoff as _kcore
        _rcore._RUST_WILCOXON = None
        _kcore._RUST_KNOCKOFF_THRESHOLD = None
        counts_gxc = (
            adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        ).T
        result = _rcore.find_clusters_recall(
            counts_gxc, resolution_start=0.6, fdr=0.05,
            max_iterations=10, seed=0, verbose=False,
        )
        adata.obs["cluster_recall"] = pd.Categorical(result.labels.astype(str))
        print(
            f"  [scop] recall(pure-py) r={result.resolution:.3f}, "
            f"{result.n_iterations} iters, "
            f"{len(np.unique(result.labels))} clusters"
        )
    except Exception as e:
        print(f"  [scop] recall failed: {type(e).__name__}: {e}")
        adata.obs["cluster_recall"] = adata.obs["cluster"]
    phases["recall"] = time.perf_counter() - t

    phases["TOTAL"] = time.perf_counter() - t_total
    return phases


def run_optimized(adata: ad.AnnData) -> dict:
    """scatlas single-sample: Rust PCA + standard kNN + Rust UMAP + Leiden
    + Rust recall.
    """
    import scanpy as sc
    from scatlas import ext, pp, tl

    t_total = time.perf_counter()
    phases: dict[str, float] = {}

    # lognorm (scipy, same as SCOP)
    t = time.perf_counter()
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((1e4 / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    adata.X = Xn.tocsr()
    phases["lognorm"] = time.perf_counter() - t

    # scatlas PCA with auto n_comps (Gavish-Donoho)
    t = time.perf_counter()
    pp.pca(adata, n_comps="auto", random_state=0)
    phases["pca"] = time.perf_counter() - t

    # Single-sample: no BBKNN, use scanpy standard kNN on scatlas PCA output
    t = time.perf_counter()
    sc.pp.neighbors(
        adata, n_neighbors=20, use_rep="X_pca",
        metric="cosine", random_state=0,
    )
    # Expose under bbknn_* keys so scatlas.tl.umap can consume them
    adata.obsp["bbknn_connectivities"] = adata.obsp["connectivities"]
    adata.obsp["bbknn_distances"] = adata.obsp["distances"]
    phases["neighbors"] = time.perf_counter() - t

    # scatlas UMAP (Hogwild rayon SGD) — init='spectral' is the new default
    # and the only init that avoids dimensional collapse on lineage data.
    t = time.perf_counter()
    tl.umap(
        adata, neighbors_key="bbknn", init="spectral",
        min_dist=0.3, spread=1.0,
        random_state=0, n_epochs=200,
    )
    phases["umap"] = time.perf_counter() - t

    # Leiden — igraph via scanpy (same fast path SCOP-aligned uses)
    t = time.perf_counter()
    sc.tl.leiden(
        adata, resolution=0.6, flavor="igraph",
        directed=False, n_iterations=2, random_state=0,
    )
    adata.obs["cluster"] = adata.obs["leiden"]
    phases["cluster"] = time.perf_counter() - t
    print(f"  [opt ] leiden → {adata.obs['cluster'].nunique()} clusters")

    # Rust recall
    t = time.perf_counter()
    try:
        import scvalidate_rust
        from scvalidate.recall_py import core as _rcore, knockoff as _kcore
        _rcore._RUST_WILCOXON = scvalidate_rust.wilcoxon_ranksum_matrix
        if hasattr(scvalidate_rust, "knockoff_threshold_offset1"):
            _kcore._RUST_KNOCKOFF_THRESHOLD = scvalidate_rust.knockoff_threshold_offset1
        counts_gxc = (
            adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        ).T
        result = _rcore.find_clusters_recall(
            counts_gxc, resolution_start=0.6, fdr=0.05,
            max_iterations=10, seed=0, verbose=False,
        )
        adata.obs["cluster_recall"] = pd.Categorical(result.labels.astype(str))
        print(
            f"  [opt ] recall(Rust) r={result.resolution:.3f}, "
            f"{result.n_iterations} iters, "
            f"{len(np.unique(result.labels))} clusters"
        )
    except Exception as e:
        print(f"  [opt ] recall failed: {type(e).__name__}: {e}")
        adata.obs["cluster_recall"] = adata.obs["cluster"]
    phases["recall"] = time.perf_counter() - t

    phases["TOTAL"] = time.perf_counter() - t_total
    return phases


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_umap(adata: ad.AnnData, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    emb = adata.obsm["X_umap"]

    panels = [
        ("CellType", "ground truth — CellType (5)"),
        ("SubCellType", "ground truth — SubCellType (8)"),
        ("cluster", "pipeline clustering (Louvain/Leiden)"),
        ("cluster_recall", "recall-validated clustering"),
    ]
    for ax, (col, subtitle) in zip(axes, panels):
        if col not in adata.obs.columns:
            ax.text(0.5, 0.5, f"{col} missing", ha="center", va="center")
            ax.set_axis_off()
            continue
        labels = adata.obs[col].astype(str).to_numpy()
        uniq = np.unique(labels)
        cmap = plt.colormaps["tab20"] if len(uniq) <= 20 else plt.colormaps["gist_ncar"]
        for i, lab in enumerate(uniq):
            mask = labels == lab
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                s=6, alpha=0.75,
                color=cmap(i / max(len(uniq) - 1, 1)),
                label=lab if len(uniq) <= 15 else None,
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(subtitle, fontsize=10)
        if len(uniq) <= 15:
            ax.legend(loc="best", fontsize=7, frameon=False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    from sklearn.metrics import adjusted_rand_score

    print("=" * 80)
    print("pancreas_sub — SCOP standard_scop (单样本) 对比")
    print("  SCOP-aligned (scanpy + pure-Py recall)  vs  scatlas-optimized (Rust recall)")
    print("=" * 80)

    adata_raw = load_pancreas_sub()
    print(f"loaded: {adata_raw.n_obs} cells × {adata_raw.n_vars} genes")
    print(f"  CellType ({adata_raw.obs['CellType'].nunique()}): "
          f"{adata_raw.obs['CellType'].cat.categories.tolist()}")
    print(f"  SubCellType ({adata_raw.obs['SubCellType'].nunique()}): "
          f"{adata_raw.obs['SubCellType'].cat.categories.tolist()}")

    print("\n" + "-" * 80)
    print("1. SCOP-aligned — scanpy + SCOP defaults + pure-Py recall")
    print("-" * 80)
    ad_scop = adata_raw.copy()
    scop_timings = run_scop_aligned(ad_scop)
    plot_umap(
        ad_scop, OUT_DIR / "pancreas_1_scop_aligned_umap.png",
        "pancreas_sub — SCOP-aligned (scanpy + Louvain 0.6 + pure-Py recall)",
    )

    print("\n" + "-" * 80)
    print("2. scatlas-optimized — Rust PCA + Rust UMAP + Rust recall")
    print("-" * 80)
    ad_opt = adata_raw.copy()
    opt_timings = run_optimized(ad_opt)
    plot_umap(
        ad_opt, OUT_DIR / "pancreas_2_optimized_umap.png",
        "pancreas_sub — scatlas-optimized (Rust PCA/UMAP/recall + Leiden 0.6)",
    )

    # Timing table
    print("\n" + "=" * 80)
    print("TIMING COMPARISON")
    print("=" * 80)
    hdr = f"{'phase':<12s} {'SCOP-align':>12s} {'optimized':>12s} {'speedup':>10s}"
    print(hdr)
    print("-" * len(hdr))
    phases = sorted(set(scop_timings) | set(opt_timings))
    for p in phases:
        s = scop_timings.get(p)
        n = opt_timings.get(p)
        s_s = f"{s:.2f}s" if s is not None else "—"
        n_s = f"{n:.2f}s" if n is not None else "—"
        sp_s = f"{s / n:.2f}×" if (s and n and n > 0) else "—"
        print(f"{p:<12s} {s_s:>12s} {n_s:>12s} {sp_s:>10s}")

    # ARI at both label levels
    print("\n" + "=" * 80)
    print("CLUSTER QUALITY (ARI vs ground truth)")
    print("=" * 80)
    gt_ct = adata_raw.obs["CellType"].astype(str).to_numpy()
    gt_sct = adata_raw.obs["SubCellType"].astype(str).to_numpy()
    for name, ad_x in [("SCOP-align", ad_scop), ("optimized", ad_opt)]:
        for col in ("cluster", "cluster_recall"):
            if col in ad_x.obs.columns:
                lbl = ad_x.obs[col].astype(str).to_numpy()
                ari_ct = adjusted_rand_score(gt_ct, lbl)
                ari_sct = adjusted_rand_score(gt_sct, lbl)
                n_cl = len(np.unique(lbl))
                print(
                    f"  {name:<11s} {col:<18s} ARI(CellType)={ari_ct:.3f}  "
                    f"ARI(SubCellType)={ari_sct:.3f}  ({n_cl} clusters)"
                )

    print("\n" + "=" * 80)
    for f in sorted(OUT_DIR.glob("pancreas_*_umap.png")):
        print(f"UMAP plot: {f}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Head-to-head: panc8_sub (1600 cells × 12940 genes, 5 techs) — 3-way

  1. Baseline    = pure scanpy, NO integration (shows the 5-tech batch effect)
  2. SCOP-aligned = scanpy + Harmony + SCOP-default UMAP (cosine, min_dist=0.3,
                    n_neighbors=30) + Louvain res=0.6 + pure-Python recall
  3. Optimized    = scatlas_pipeline (BBKNN + Harmony + Rust UMAP) + Rust recall

All three produce UMAPs with celltype / cluster / recall-cluster panels.
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

PANC8_RDA = Path("/mnt/f/NMF_rewrite/panc8_sub.rda")
OUT_DIR = Path(__file__).resolve().parent / "panc8_results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def load_panc8() -> ad.AnnData:
    """Seurat rds → AnnData (cells × genes)."""
    import rdata

    parsed = rdata.read_rda(str(PANC8_RDA))
    seurat = parsed["panc8_sub"]
    rna = seurat.assays["RNA"]
    counts_raw = rna.layers["counts"]
    n_genes, n_cells = int(counts_raw.Dim[0]), int(counts_raw.Dim[1])
    mat_gc = sp.csc_matrix(
        (counts_raw.x, counts_raw.i, counts_raw.p),
        shape=(n_genes, n_cells),
    )
    # AnnData expects cells × genes
    X = mat_gc.T.tocsr().astype(np.float32)

    meta = getattr(seurat, "meta_data", None) or getattr(seurat, "meta.data", None)
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    for col in meta.columns:
        adata.obs[str(col)] = pd.Categorical(meta[col].to_numpy())
    return adata


# ---------------------------------------------------------------------------
# Original pipeline (pure Python / pre-Rust baseline)
# ---------------------------------------------------------------------------


def run_baseline_scanpy(adata: ad.AnnData) -> dict:
    """Pure scanpy pipeline, NO integration — baseline showing the
    uncorrected batch effect.
    """
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
    sc.tl.pca(ad_hvg, n_comps=30, random_state=0)
    adata.obsm["X_pca"] = ad_hvg.obsm["X_pca"]
    phases["pca"] = time.perf_counter() - t

    t = time.perf_counter()
    sc.pp.neighbors(adata, n_pcs=30, random_state=0)
    phases["neighbors"] = time.perf_counter() - t

    t = time.perf_counter()
    sc.tl.umap(adata, random_state=0)
    phases["umap"] = time.perf_counter() - t

    t = time.perf_counter()
    sc.tl.leiden(
        adata, resolution=0.8, flavor="igraph",
        directed=False, n_iterations=2, random_state=0,
    )
    phases["leiden"] = time.perf_counter() - t
    print(f"  [base] leiden → {adata.obs['leiden'].nunique()} clusters")
    # Baseline doesn't run recall; recall is expensive and the batch effect
    # makes cluster assignment meaningless on the uncorrected pipeline.
    adata.obs["leiden_recall"] = adata.obs["leiden"]
    phases["TOTAL"] = time.perf_counter() - t_total
    return phases


def run_scop_aligned(adata: ad.AnnData) -> dict:
    """SCOP-style pipeline: scanpy + Harmony + UMAP (cosine, min_dist=0.3,
    n_neighbors=30) + Louvain res=0.6 + pure-Python recall.

    Mirrors what a user following SCOP conventions in Python would write.
    Uses scatlas.ext.harmony as the Harmony impl (faithful port of the R
    package, same defaults) so the SCOP speed difference comes from
    workflow (not Harmony implementation).
    """
    import scanpy as sc
    from scatlas import ext

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
    sc.tl.pca(ad_hvg, n_comps=50, random_state=0)    # SCOP: 50 PCs
    adata.obsm["X_pca"] = ad_hvg.obsm["X_pca"]
    phases["pca"] = time.perf_counter() - t

    t = time.perf_counter()
    adata.obs["_batch"] = adata.obs["tech"].astype(str)
    ext.harmony(
        adata, batch_key="_batch", use_rep="X_pca",
        n_clusters=min(50, adata.n_obs // 30), seed=0,
    )
    phases["harmony"] = time.perf_counter() - t

    # Standard kNN graph on Harmony embedding (no BBKNN)
    t = time.perf_counter()
    sc.pp.neighbors(
        adata, n_neighbors=30, use_rep="X_pca_harmony",
        metric="cosine", random_state=0,
    )
    phases["neighbors"] = time.perf_counter() - t

    # SCOP UMAP: min_dist=0.3 (tighter than scanpy 0.5), cosine
    t = time.perf_counter()
    sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=0)
    phases["umap"] = time.perf_counter() - t

    # SCOP clustering: Louvain resolution=0.6
    t = time.perf_counter()
    sc.tl.louvain(adata, resolution=0.6, random_state=0, flavor="igraph", directed=False)
    adata.obs["leiden"] = adata.obs["louvain"]     # alias for downstream column check
    phases["cluster"] = time.perf_counter() - t
    print(f"  [scop] louvain → {adata.obs['leiden'].nunique()} clusters")

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
        adata.obs["leiden_recall"] = pd.Categorical(result.labels.astype(str))
        print(
            f"  [scop] recall(pure-py) r={result.resolution:.3f}, "
            f"{result.n_iterations} iters, "
            f"{len(np.unique(result.labels))} clusters"
        )
    except Exception as e:
        print(f"  [scop] recall failed: {type(e).__name__}: {e}")
        adata.obs["leiden_recall"] = adata.obs["leiden"]
    phases["recall"] = time.perf_counter() - t

    phases["TOTAL"] = time.perf_counter() - t_total
    return phases


# ---------------------------------------------------------------------------
# Optimized pipeline (scatlas + Rust recall)
# ---------------------------------------------------------------------------


def run_optimized(adata: ad.AnnData) -> dict:
    """scatlas_pipeline end-to-end + scvalidate recall with Rust wilcoxon."""
    from scatlas import ext, metrics, pp, tl

    t_total = time.perf_counter()
    phases: dict[str, float] = {}

    # lognorm
    t = time.perf_counter()
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((1e4 / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    adata.X = Xn.tocsr()
    phases["lognorm"] = time.perf_counter() - t

    # pca (Rust)
    t = time.perf_counter()
    pp.pca(adata, n_comps="auto", random_state=0)
    phases["pca"] = time.perf_counter() - t

    # bbknn (Rust HNSW + Rust fuzzy)
    t = time.perf_counter()
    adata.obs["_batch"] = adata.obs["tech"].astype(str)
    ext.bbknn(
        adata, batch_key="_batch", use_rep="X_pca",
        neighbors_within_batch=3, with_connectivities=True,
    )
    phases["bbknn"] = time.perf_counter() - t

    # harmony (Rust)
    t = time.perf_counter()
    ext.harmony(
        adata, batch_key="_batch", use_rep="X_pca",
        n_clusters=min(50, adata.n_obs // 30), seed=0,
    )
    phases["harmony"] = time.perf_counter() - t

    # umap (Rust Hogwild)
    t = time.perf_counter()
    tl.umap(adata, neighbors_key="bbknn", init="pca", random_state=0, n_epochs=200)
    phases["umap"] = time.perf_counter() - t

    # leiden
    t = time.perf_counter()
    import scanpy as sc
    adata.obsp["connectivities"] = adata.obsp["bbknn_connectivities"]
    adata.obsp["distances"] = adata.obsp["bbknn_distances"]
    adata.uns["neighbors"] = {
        "params": {"method": "umap", "n_neighbors": 6},
        "connectivities_key": "connectivities",
        "distances_key": "distances",
    }
    sc.tl.leiden(
        adata, resolution=0.8, flavor="igraph",
        directed=False, n_iterations=2, random_state=0,
    )
    phases["leiden"] = time.perf_counter() - t
    print(f"  [opt ] leiden → {adata.obs['leiden'].nunique()} clusters")

    # Recall (Rust wilcoxon)
    t = time.perf_counter()
    try:
        # Re-enable Rust paths (may have been turned off in original run)
        import scvalidate_rust
        from scvalidate.recall_py import core as _rcore, knockoff as _kcore
        _rcore._RUST_WILCOXON = scvalidate_rust.wilcoxon_ranksum_matrix
        if hasattr(scvalidate_rust, "knockoff_threshold_offset1"):
            _kcore._RUST_KNOCKOFF_THRESHOLD = scvalidate_rust.knockoff_threshold_offset1
        counts_gxc = (
            adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        ).T
        result = _rcore.find_clusters_recall(
            counts_gxc, resolution_start=0.8, fdr=0.05,
            max_iterations=10, seed=0, verbose=False,
        )
        adata.obs["leiden_recall"] = pd.Categorical(result.labels.astype(str))
        print(
            f"  [opt ] recall converged r={result.resolution:.3f}, "
            f"{result.n_iterations} iters, "
            f"{len(np.unique(result.labels))} clusters"
        )
    except Exception as e:
        print(f"  [opt ] recall failed: {type(e).__name__}: {e}")
        adata.obs["leiden_recall"] = adata.obs["leiden"]
    phases["recall"] = time.perf_counter() - t

    # scib metrics
    t = time.perf_counter()
    raw = ext.bbknn_kneighbors(
        adata.obsm.get("X_pca_harmony", adata.obsm["X_pca"]),
        adata.obs["_batch"].to_numpy(),
        neighbors_within_batch=3, backend="brute",
    )
    idx = raw["indices"].astype(np.int32)
    dist = raw["distances"].astype(np.float32)
    sent = raw["indices"] == np.iinfo(np.uint32).max
    if sent.any():
        rowi = np.broadcast_to(
            np.arange(idx.shape[0], dtype=np.int32)[:, None], idx.shape
        )
        idx = np.where(sent, rowi, idx)
        dist[sent] = 0.0
    scib = metrics.scib_score(
        idx, dist,
        batch_labels=adata.obs["_batch"].to_numpy(),
        label_labels=adata.obs["celltype"].astype(str).to_numpy(),
    )
    for k, v in scib.items():
        if isinstance(v, float):
            print(f"             scib {k:22s} = {v:.3f}")
    phases["scib"] = time.perf_counter() - t
    phases["TOTAL"] = time.perf_counter() - t_total
    return phases


# ---------------------------------------------------------------------------
# UMAP plotter
# ---------------------------------------------------------------------------


def plot_umap(adata: ad.AnnData, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    emb = adata.obsm["X_umap"]

    for ax, col, subtitle in [
        (axes[0], "celltype", "celltype (ground truth)"),
        (axes[1], "leiden", "leiden (pipeline clustering)"),
        (axes[2], "leiden_recall", "recall-validated clustering"),
    ]:
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
                s=3, alpha=0.6, color=cmap(i / max(len(uniq) - 1, 1)),
                label=lab if len(uniq) <= 15 else None,
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(subtitle)
        if len(uniq) <= 15:
            ax.legend(loc="best", fontsize=7, frameon=False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    from sklearn.metrics import adjusted_rand_score

    print("=" * 72)
    print("panc8_sub 对比:原始 (scanpy + 纯 Python recall) vs 优化 (scatlas + Rust recall)")
    print("=" * 72)

    adata_raw = load_panc8()
    print(f"loaded: {adata_raw.n_obs} cells × {adata_raw.n_vars} genes")
    print(f"  techs: {adata_raw.obs['tech'].cat.categories.tolist()}")
    print(f"  celltypes: {len(adata_raw.obs['celltype'].cat.categories)} "
          f"({adata_raw.obs['celltype'].cat.categories.tolist()})")

    # --- 3-way comparison ---
    print("\n" + "-" * 72)
    print("1. BASELINE SCANPY (no integration — shows batch effect)")
    print("-" * 72)
    ad_base = adata_raw.copy()
    base_timings = run_baseline_scanpy(ad_base)
    plot_umap(ad_base, OUT_DIR / "panc8_1_baseline_umap.png",
              "panc8 — baseline scanpy (no Harmony)")

    print("\n" + "-" * 72)
    print("2. SCOP-ALIGNED (scanpy + Harmony + cosine UMAP + Louvain 0.6 + pure-Py recall)")
    print("-" * 72)
    ad_scop = adata_raw.copy()
    scop_timings = run_scop_aligned(ad_scop)
    plot_umap(ad_scop, OUT_DIR / "panc8_2_scop_aligned_umap.png",
              "panc8 — SCOP-aligned (scanpy + Harmony, cosine UMAP, Louvain 0.6)")

    print("\n" + "-" * 72)
    print("3. OPTIMIZED (scatlas_pipeline + Rust wilcoxon)")
    print("-" * 72)
    ad_opt = adata_raw.copy()
    opt_timings = run_optimized(ad_opt)
    plot_umap(ad_opt, OUT_DIR / "panc8_3_optimized_umap.png",
              "panc8 — scatlas_pipeline (BBKNN + Harmony + Rust UMAP/recall)")

    # Alias old names for the table
    orig_timings = base_timings

    # Timing table
    print("\n" + "=" * 80)
    print("TIMING COMPARISON")
    print("=" * 80)
    header = f"{'phase':<12s} {'baseline':>12s} {'SCOP-align':>12s} {'optimized':>12s} {'opt vs SCOP':>12s}"
    print(header)
    print("-" * len(header))
    all_phases = sorted(set(base_timings) | set(scop_timings) | set(opt_timings))
    for p in all_phases:
        b = base_timings.get(p)
        s = scop_timings.get(p)
        n = opt_timings.get(p)
        b_s = f"{b:.2f}s" if b is not None else "—"
        s_s = f"{s:.2f}s" if s is not None else "—"
        n_s = f"{n:.2f}s" if n is not None else "—"
        sp_s = f"{s / n:.2f}×" if (s and n and n > 0) else "—"
        print(f"{p:<12s} {b_s:>12s} {s_s:>12s} {n_s:>12s} {sp_s:>12s}")

    # Cluster agreement vs ground truth
    print("\n" + "=" * 80)
    print("CLUSTER QUALITY (ARI vs celltype ground truth)")
    print("=" * 80)
    gt = adata_raw.obs["celltype"].astype(str).to_numpy()
    for name, ad_x in [("baseline", ad_base), ("SCOP-align", ad_scop), ("optimized", ad_opt)]:
        for col in ("leiden", "leiden_recall"):
            if col in ad_x.obs.columns:
                labels = ad_x.obs[col].astype(str).to_numpy()
                ari = adjusted_rand_score(gt, labels)
                print(f"  {name:<11s} {col:<18s} ARI={ari:.3f}  "
                      f"({len(np.unique(labels))} clusters)")

    print("\n" + "=" * 80)
    for f in sorted(OUT_DIR.glob("panc8_*_umap.png")):
        print(f"UMAP plot: {f}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

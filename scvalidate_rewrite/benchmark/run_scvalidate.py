"""Run scvalidate on the pancreas_sub developmental benchmark.

Inputs (exported from pancreas_sub.rda by export.R):
    counts.mtx, genes.tsv, cells.tsv, metadata.csv
    (1000 cells, 15998 genes, CellType=5 / SubCellType=8, single batch)

Outputs (written to benchmark/):
    scvalidate_clusters.csv       — recall + scSHC labels + verdicts per cell
    scvalidate_report.csv         — fuse layer per-cluster report
    scvalidate_timing.json        — wall-clock times for each module
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.io import mmread

from scvalidate import (
    AutoClusterReport,
    calculate_rogue,
    find_clusters_recall,
    fuse_report,
    scshc,
    se_fun,
    test_scshc_clusters,
)


BENCH = Path("F:/NMF_rewrite/scvalidate_rewrite/benchmark")


def load_dataset() -> AnnData:
    """Load exported panc8 into an AnnData (cells × genes)."""
    counts = mmread(BENCH / "counts.mtx").tocsr()  # genes × cells
    genes = (BENCH / "genes.tsv").read_text().strip().split("\n")
    cells = (BENCH / "cells.tsv").read_text().strip().split("\n")
    meta = pd.read_csv(BENCH / "metadata.csv", index_col=0)

    # Keep cells×genes orientation for AnnData
    adata = AnnData(X=counts.T.tocsr())
    adata.var_names = genes
    adata.obs_names = cells
    adata.obs = meta.loc[cells]
    return adata


def run_baseline_clustering(adata: AnnData, seed: int = 0) -> AnnData:
    """Standard scanpy Leiden clustering for sc-SHC post-hoc input + markers."""
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat")
    ad = ad[:, ad.var["highly_variable"]].copy()
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=30, random_state=seed)
    sc.pp.neighbors(ad, n_pcs=30, random_state=seed)
    sc.tl.leiden(
        ad, resolution=0.8, random_state=seed,
        flavor="igraph", n_iterations=2, directed=False,
    )
    adata.obs["leiden"] = ad.obs["leiden"].values
    adata.obsm["X_pca"] = ad.obsm["X_pca"]
    return adata


def main() -> None:
    timing: dict[str, float] = {}

    print("[1/5] Loading pancreas_sub …")
    t0 = time.time()
    adata = load_dataset()
    timing["load"] = time.time() - t0
    print(f"  shape: {adata.shape}, "
          f"CellType: {adata.obs['CellType'].nunique()}, "
          f"SubCellType: {adata.obs['SubCellType'].nunique()}")

    print("[2/5] Baseline Leiden clustering …")
    t0 = time.time()
    adata = run_baseline_clustering(adata, seed=0)
    timing["baseline_leiden"] = time.time() - t0
    n_leiden = adata.obs["leiden"].nunique()
    print(f"  n_leiden_clusters: {n_leiden}")

    # Genes × cells raw counts for the method ports
    counts_gxc = np.asarray(adata.X.T.todense())

    print("[3/5] Running recall (knockoff + resolution search) …")
    t0 = time.time()
    recall_res = find_clusters_recall(
        counts_gxc,
        resolution_start=0.8,
        reduction_percentage=0.2,
        dims=30,
        null_method="ZIP",
        n_variable_features=2000,
        fdr=0.05,
        max_iterations=6,
        seed=0,
        verbose=True,
    )
    timing["recall"] = time.time() - t0
    adata.obs["recall_cluster"] = recall_res.labels
    print(f"  recall: n_clusters={len(np.unique(recall_res.labels))}, "
          f"iter={recall_res.n_iterations}, final_res={recall_res.resolution:.3f}")

    print("[4/5] Running sc-SHC testClusters on baseline Leiden …")
    t0 = time.time()
    # pancreas_sub is a single batch; no batch covariate.
    new_labels, _tree, scshc_pvalues = test_scshc_clusters(
        counts_gxc,
        cluster_ids=adata.obs["leiden"].to_numpy(),
        batch=None,
        alpha=0.05,
        num_features=2000,
        num_pcs=30,
        seed=0,
    )
    timing["scshc"] = time.time() - t0
    adata.obs["scshc_merged"] = new_labels
    n_scshc = len(np.unique(new_labels))
    print(f"  sc-SHC merged: {n_leiden} → {n_scshc} clusters")

    print("[5/5] Computing ROGUE per leiden cluster + marker counts …")
    t0 = time.time()
    from scvalidate.rogue_py.core import filter_matrix
    rogue_by_cluster: dict[int, float] = {}
    for c in sorted(adata.obs["leiden"].unique().astype(int)):
        mask = (adata.obs["leiden"].astype(int) == c).to_numpy()
        sub = counts_gxc[:, mask]
        # Match R ROGUE benchmark: matr.filter(min.cells=10, min.genes=10)
        # before SE_fun. Without this, Python ROGUE runs ~2% lower than R.
        sub_f = filter_matrix(sub, min_cells=10, min_genes=10)
        table = se_fun(sub_f, span=0.5, r=1.0)
        rogue_by_cluster[c] = float(calculate_rogue(table, platform="UMI"))

    # Marker richness — use scanpy rank_genes_groups on normalized data
    sc.tl.rank_genes_groups(
        adata, groupby="leiden", method="wilcoxon", use_raw=False,
    )
    n_markers: dict[int, int] = {}
    for c in sorted(adata.obs["leiden"].unique().astype(int)):
        df = sc.get.rank_genes_groups_df(adata, group=str(c))
        n_markers[c] = int(
            ((df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 1.0)).sum()
        )
    timing["rogue_and_markers"] = time.time() - t0

    print("[fuse] Building AutoClusterReport …")
    # Map each Leiden cluster → scSHC p-value (use the original-cluster-id key)
    scshc_p_by_leiden: dict[int, float] = {
        int(k): float(v) for k, v in scshc_pvalues.items()
    }
    # recall_pass: True if the leiden cluster still appears in recall_res.labels
    # (recall produces its own partition, so we approximate: a Leiden cluster
    # "passes recall" if it is NOT fully merged with another in recall's
    # coarser solution — measured by whether the majority of its cells share
    # a unique recall label)
    recall_pass: dict[int, bool] = {}
    leiden = adata.obs["leiden"].astype(int).to_numpy()
    for c in np.unique(leiden):
        cells_in_c = leiden == c
        recall_labs = recall_res.labels[cells_in_c]
        vals, counts = np.unique(recall_labs, return_counts=True)
        maj = vals[np.argmax(counts)]
        # pass if ≥80% of cells stayed within a dedicated recall label
        recall_pass[int(c)] = bool(counts.max() / counts.sum() >= 0.8)

    cluster_ids = sorted(int(c) for c in np.unique(leiden))
    report: AutoClusterReport = fuse_report(
        cluster_ids,
        recall_pass=recall_pass,
        scshc_pvalue=scshc_p_by_leiden,
        rogue_score=rogue_by_cluster,
        n_markers=n_markers,
        alpha=0.05,
    )

    # Per-cluster CSV
    report.to_dataframe().to_csv(BENCH / "scvalidate_report.csv", index=False)
    # Per-cell CSV
    adata.obs[["CellType", "SubCellType", "Phase",
               "leiden", "recall_cluster", "scshc_merged"]].to_csv(
        BENCH / "scvalidate_clusters.csv"
    )
    with open(BENCH / "scvalidate_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    total = sum(timing.values())
    print(f"\n=== done ===")
    print(f"  total wall-clock: {total:.1f}s")
    for k, v in timing.items():
        print(f"    {k:25s} {v:6.1f}s")
    print(f"\nReport head:")
    print(report.to_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()

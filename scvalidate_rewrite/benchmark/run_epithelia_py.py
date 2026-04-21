"""scvalidate end-to-end on a stratified subsample of seurat_epithelia.

Inputs (benchmark/epithelia/ exported by export_epithelia.R):
    counts.mtx, genes.tsv, cells.tsv, metadata.csv

Outputs (benchmark/epithelia/):
    subsample_cells.txt           — cell barcodes used (row-aligned with python+R)
    py_report.csv                 — fuse layer per-cluster report
    py_clusters.csv               — per-cell labels
    py_timing.json                — wall clocks
    py_run.log                    — tee of stdout
"""
from __future__ import annotations

import json
import sys
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
    se_fun,
    test_scshc_clusters,
)
from scvalidate.rogue_py.core import filter_matrix


BENCH = Path("F:/NMF_rewrite/scvalidate_rewrite/benchmark/epithelia")

# Subsample size. 10k cells keeps recall's augmented dense mat at ~20k x 10k x 8B
# = 1.6GB, which fits comfortably in 64GB.
SUBSAMPLE_N = 10_000
STRATIFY_COL = "subtype"  # 3+ biologically distinct groups inside Epithelia
SEED = 0


def log(msg: str) -> None:
    print(msg, flush=True)


def load_and_subsample() -> tuple[AnnData, list[str]]:
    log("[load] reading mtx + meta ...")
    t0 = time.time()
    counts = mmread(BENCH / "counts.mtx").tocsr()  # genes x cells
    genes = (BENCH / "genes.tsv").read_text().strip().split("\n")
    cells = (BENCH / "cells.tsv").read_text().strip().split("\n")
    meta = pd.read_csv(BENCH / "metadata.csv", index_col=0)
    meta = meta.loc[cells]
    log(f"  loaded {counts.shape[1]} cells × {counts.shape[0]} genes "
        f"in {time.time()-t0:.1f}s")

    # Stratified subsample
    log(f"[subsample] target {SUBSAMPLE_N}, stratify by {STRATIFY_COL}")
    rng = np.random.default_rng(SEED)
    strata = meta[STRATIFY_COL].astype(str).fillna("NA")
    counts_per_stratum = strata.value_counts()
    log(f"  strata: {counts_per_stratum.to_dict()}")
    # Proportional allocation, min 50 per stratum to avoid degenerate clusters
    alloc = {}
    remaining = SUBSAMPLE_N
    for k, n in counts_per_stratum.items():
        want = max(50, int(round(SUBSAMPLE_N * n / counts_per_stratum.sum())))
        alloc[k] = min(want, n)
    # Rescale if overallocated
    total = sum(alloc.values())
    if total > SUBSAMPLE_N:
        scale = SUBSAMPLE_N / total
        alloc = {k: int(v * scale) for k, v in alloc.items()}
    log(f"  alloc: {alloc}")

    picked = []
    for k, n in alloc.items():
        idx = np.where(strata.values == k)[0]
        if len(idx) <= n:
            picked.extend(idx.tolist())
        else:
            picked.extend(rng.choice(idx, size=n, replace=False).tolist())
    picked = sorted(picked)
    log(f"  picked {len(picked)} cells")

    # Build AnnData: cells × genes
    sub_counts = counts[:, picked]  # genes x subset
    cell_ids = [cells[i] for i in picked]
    sub_meta = meta.iloc[picked].copy()
    adata = AnnData(X=sub_counts.T.tocsr())
    adata.var_names = genes
    adata.obs_names = cell_ids
    adata.obs = sub_meta

    # Save subsample cell ids for R matching
    (BENCH / "subsample_cells.txt").write_text("\n".join(cell_ids))
    return adata, cell_ids


def baseline_leiden(adata: AnnData, seed: int = SEED) -> AnnData:
    """scanpy leiden clustering for scSHC + markers."""
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

    log("=" * 60)
    log("scvalidate on seurat_epithelia (subsampled)")
    log("=" * 60)

    t0 = time.time()
    adata, cell_ids = load_and_subsample()
    timing["load_subsample"] = time.time() - t0

    t0 = time.time()
    log("[leiden] baseline scanpy clustering ...")
    adata = baseline_leiden(adata)
    timing["baseline_leiden"] = time.time() - t0
    n_leiden = adata.obs["leiden"].nunique()
    log(f"  n_leiden = {n_leiden}")

    # For algorithmic ports: genes × cells dense. At 10k cells x 16k genes,
    # that's 1.2 GB — fits fine.
    counts_gxc = np.asarray(adata.X.T.todense())
    log(f"  counts_gxc dense: {counts_gxc.shape}, "
        f"{counts_gxc.nbytes / 1e9:.2f} GB")

    # --- recall ---
    log("[recall] ...")
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
        seed=SEED,
        verbose=True,
    )
    timing["recall"] = time.time() - t0
    adata.obs["recall_cluster"] = recall_res.labels
    log(f"  recall n_clusters={len(np.unique(recall_res.labels))}, "
        f"iter={recall_res.n_iterations}, res={recall_res.resolution:.3f}")

    # --- scSHC ---
    log("[scshc] testClusters on leiden ...")
    t0 = time.time()
    new_labels, _tree, scshc_pvalues = test_scshc_clusters(
        counts_gxc,
        cluster_ids=adata.obs["leiden"].to_numpy(),
        batch=None,
        alpha=0.05,
        num_features=2000,
        num_pcs=30,
        seed=SEED,
    )
    timing["scshc"] = time.time() - t0
    adata.obs["scshc_merged"] = new_labels
    log(f"  scSHC merged: {n_leiden} -> {len(np.unique(new_labels))}")

    # --- ROGUE per leiden cluster ---
    log("[rogue] per-leiden ROGUE ...")
    t0 = time.time()
    rogue_by_cluster: dict[int, float] = {}
    for c in sorted(adata.obs["leiden"].unique().astype(int)):
        mask = (adata.obs["leiden"].astype(int) == c).to_numpy()
        sub = counts_gxc[:, mask]
        sub_f = filter_matrix(sub, min_cells=10, min_genes=10)
        if sub_f.shape[0] < 50 or sub_f.shape[1] < 10:
            log(f"    cluster {c}: too few surviving genes/cells, skip")
            rogue_by_cluster[c] = float("nan")
            continue
        table = se_fun(sub_f, span=0.5, r=1.0)
        rogue_by_cluster[c] = float(calculate_rogue(table, platform="UMI"))
    # Markers
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

    # --- fuse ---
    log("[fuse] building AutoClusterReport ...")
    scshc_p_by_leiden = {int(k): float(v) for k, v in scshc_pvalues.items()}

    leiden = adata.obs["leiden"].astype(int).to_numpy()
    recall_pass: dict[int, bool] = {}
    for c in np.unique(leiden):
        mask = leiden == c
        rl = recall_res.labels[mask]
        _, counts_ = np.unique(rl, return_counts=True)
        recall_pass[int(c)] = bool(counts_.max() / counts_.sum() >= 0.8)

    cluster_ids = sorted(int(c) for c in np.unique(leiden))
    report: AutoClusterReport = fuse_report(
        cluster_ids,
        recall_pass=recall_pass,
        scshc_pvalue=scshc_p_by_leiden,
        rogue_score=rogue_by_cluster,
        n_markers=n_markers,
        alpha=0.05,
    )
    report.to_dataframe().to_csv(BENCH / "py_report.csv", index=False)

    keep_cols = [c for c in ("subtype", "coarse_ano", "patient", "sample",
                              "data_sets", "leiden", "recall_cluster",
                              "scshc_merged") if c in adata.obs.columns]
    adata.obs[keep_cols].to_csv(BENCH / "py_clusters.csv")

    with open(BENCH / "py_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    log("\n=== done ===")
    log(f"  total = {sum(timing.values()):.1f}s")
    for k, v in timing.items():
        log(f"    {k:22s} {v:7.1f}s")
    log("\nReport head:")
    log(report.to_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()

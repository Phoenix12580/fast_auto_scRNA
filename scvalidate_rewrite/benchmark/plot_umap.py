"""Generate UMAP panels showing:
    1. Ground-truth cell type
    2. Batch (tech)
    3. Baseline Leiden clusters
    4. scvalidate verdict (HIGH/MED/LOW/REJECT)
    5. scSHC merged labels
    6. recall labels

Reads the outputs of run_scvalidate.py and the raw counts; writes PNG to
benchmark/scvalidate_umap.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.io import mmread


BENCH = Path("F:/NMF_rewrite/scvalidate_rewrite/benchmark")


def main() -> None:
    counts = mmread(BENCH / "counts.mtx").tocsr()
    genes = (BENCH / "genes.tsv").read_text().strip().split("\n")
    cells = (BENCH / "cells.tsv").read_text().strip().split("\n")
    meta = pd.read_csv(BENCH / "metadata.csv", index_col=0).loc[cells]
    cluster_df = pd.read_csv(BENCH / "scvalidate_clusters.csv", index_col=0).loc[cells]
    report_df = pd.read_csv(BENCH / "scvalidate_report.csv")

    adata = AnnData(X=counts.T.tocsr())
    adata.var_names = genes
    adata.obs_names = cells
    adata.obs = meta.join(
        cluster_df.drop(columns=["CellType", "SubCellType", "Phase"])
    )

    # Standard scanpy pipeline for UMAP
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30, random_state=0)
    sc.pp.neighbors(adata, n_pcs=30, random_state=0)
    sc.tl.umap(adata, random_state=0)

    # Map Leiden cluster → verdict
    verdict_by_cluster = dict(zip(report_df["cluster_id"], report_df["verdict"]))
    adata.obs["verdict"] = (
        adata.obs["leiden"].astype(int).map(verdict_by_cluster).astype("category")
    )
    adata.obs["leiden"] = adata.obs["leiden"].astype(str).astype("category")
    adata.obs["recall_cluster"] = (
        adata.obs["recall_cluster"].astype(str).astype("category")
    )
    adata.obs["scshc_merged"] = adata.obs["scshc_merged"].astype("category")

    # 2 × 3 panel
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    panels = [
        ("CellType", "1. Ground truth CellType (5)"),
        ("SubCellType", "2. Ground truth SubCellType (8)"),
        ("leiden", "3. Baseline Leiden clusters"),
        ("verdict", "4. scvalidate verdict"),
        ("scshc_merged", "5. sc-SHC merged"),
        ("recall_cluster", "6. recall clusters"),
    ]
    # Fixed color palette for verdicts so HIGH→green, MED→orange, LOW→red, REJECT→gray
    verdict_palette = {
        "HIGH": "#2ca02c",
        "MED": "#ff7f0e",
        "LOW": "#d62728",
        "REJECT": "#808080",
    }
    for ax, (col, title) in zip(axes.flat, panels):
        if col == "verdict":
            palette = [verdict_palette.get(v, "#888888")
                       for v in adata.obs["verdict"].cat.categories]
            sc.pl.umap(
                adata, color=col, ax=ax, show=False, frameon=False,
                title=title, palette=palette, legend_fontsize=9,
            )
        else:
            sc.pl.umap(
                adata, color=col, ax=ax, show=False, frameon=False,
                title=title, legend_fontsize=8,
            )

    plt.tight_layout()
    out = BENCH / "scvalidate_umap.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

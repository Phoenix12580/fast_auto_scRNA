"""pancreas_sub — integration pipeline verification via scatlas_pipeline.

Loads pancreas_sub (1000 cells × 15998 genes, pancreatic development
lineage) and runs ``scatlas_pipeline`` with ``integration="all"``. Every
step uses a Rust-backed kernel where one exists (PCA / BBKNN / Harmony /
fuzzy_set / UMAP / scIB / recall).

pancreas_sub is a **single-batch** dataset (``orig.ident='SeuratProject'``),
so the pipeline's auto-degradation kicks in: BBKNN + Harmony are skipped
(both require ≥ 2 batches) and only the ``none`` route runs. This is the
correct behavior — it verifies the pipeline gracefully handles single-
batch input. For multi-batch integration-quality testing, a separate
multi-batch dataset is needed.

Outputs:
  * ``pancreas_integration_umap.png``    — UMAP colored by CellType.
  * ``pancreas_scib_compare.csv``        — scIB metric table.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

import anndata as ad  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scatlas_pipeline.pipeline import (  # noqa: E402
    PipelineConfig, run_from_config, compare_integration_plot,
)

PANCREAS_RDA = Path("/mnt/f/NMF_rewrite/pancreas_sub.rda")
OUT_DIR = Path(__file__).resolve().parent / "pancreas_results"
OUT_DIR.mkdir(exist_ok=True)


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
    a = ad.AnnData(X=X)
    a.obs_names = [f"cell_{i}" for i in range(n_cells)]
    a.var_names = [f"gene_{i}" for i in range(n_genes)]
    for col in meta.columns:
        a.obs[str(col)] = pd.Categorical(meta[col].to_numpy())
    # pancreas_sub's orig.ident is 'SeuratProject' for every cell; normalize.
    if "orig.ident" not in a.obs.columns:
        a.obs["orig.ident"] = "SeuratProject"
    return a


def main() -> int:
    if not PANCREAS_RDA.exists():
        print(f"pancreas_sub.rda not found at {PANCREAS_RDA} — skipping")
        return 2

    adata_in = load_pancreas_sub()
    print(f"loaded: {adata_in.n_obs} × {adata_in.n_vars}")
    for col in ("orig.ident", "CellType", "SubCellType"):
        if col in adata_in.obs.columns:
            vc = adata_in.obs[col].value_counts()
            print(f"  {col:15s} ({vc.shape[0]} unique): {dict(vc.head())}")

    cfg = PipelineConfig(
        input_h5ad="<in-memory:pancreas_sub>",
        batch_key="orig.ident",
        min_cells=3, min_genes=50, max_pct_mt=100.0,
        hvg_n_top_genes=2000, hvg_flavor="seurat_v3", hvg_batch_aware=False,
        # SCOP/Seurat reference uses 50 PCs + cosine kNN @ k=30; match
        # that to compare against R CellDimPlot reference figures.
        pca_n_comps=50,
        integration="all",        # 1 batch → auto-degrades to 'none' only
        neighbors_within_batch=3,
        # knn_n_neighbors=30, knn_metric="cosine", umap_init="pca" are
        # pipeline defaults now — no need to override.
        umap_n_epochs=200, umap_min_dist=0.3,
        run_recall=True,          # 1000 cells — recall is affordable
        label_key="CellType",
    )
    adata = run_from_config(cfg, adata_in=adata_in)

    # Report which routes actually ran (single-batch auto-skips bbknn/harmony).
    from scatlas_pipeline.pipeline import INTEGRATION_METHODS
    ran = [m for m in INTEGRATION_METHODS if f"X_pca_{m}" in adata.obsm]
    print(f"\nroutes executed: {ran}")

    # --- comparison plot (celltype + batch) ---
    p1 = compare_integration_plot(
        adata, OUT_DIR / "pancreas_integration_umap.png",
        label_key="CellType",
    )
    p2 = compare_integration_plot(
        adata, OUT_DIR / "pancreas_integration_batch.png",
        label_key="_batch",
    )

    # --- scIB table (one row if only 'none' ran) ---
    rows = []
    for m in ran:
        scib = adata.uns.get(f"scib_{m}", {})
        row = {"method": m}
        for k, v in scib.items():
            if isinstance(v, (int, float)):
                row[k] = round(float(v), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv = OUT_DIR / "pancreas_scib_compare.csv"
    df.to_csv(csv, index=False)
    print("\n" + "=" * 72)
    print("scIB summary")
    print("=" * 72)
    print(df.to_string(index=False))
    print("\noutputs:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

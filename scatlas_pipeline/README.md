# scatlas_pipeline

**单样本/小-中等 atlas scRNA-seq 端到端流程 — 到聚类为止全部 Rust-优化。**

上游项目:
- [`scatlas`](../scatlas) — Rust kernels (PCA, BBKNN, Harmony, UMAP, fuzzy_simplicial_set, scib metrics)
- [`scvalidate_rewrite`](../scvalidate_rewrite) — Rust kernels (wilcoxon, knockoff, ROGUE) + `find_clusters_recall`

这个目录把两者串成一条流水线,一次调用完成 **load → QC → lognorm → PCA → BBKNN → (Harmony) → UMAP → Leiden → recall-validated clustering → scib metrics**。

## 流程

```
┌──────────────────────────────────────────────────┐
│ 01  load h5ad + QC (min_cells, n_genes, pct_mt)  │  scipy
├──────────────────────────────────────────────────┤
│ 02  lognorm (1e4 CPM + log1p)                    │  scipy
├──────────────────────────────────────────────────┤
│ 03  PCA (GD auto n_comps)                        │  ⚡ scatlas.pp.pca
├──────────────────────────────────────────────────┤
│ 04  BBKNN kneighbors + fuzzy connectivities      │  ⚡ scatlas.ext.bbknn (HNSW + Rust fuzzy)
├──────────────────────────────────────────────────┤
│ 05  Harmony 2.0 (可选)                           │  ⚡ scatlas.ext.harmony  (6× R)
├──────────────────────────────────────────────────┤
│ 06  UMAP layout (Hogwild rayon)                  │  ⚡ scatlas.tl.umap    (16.75× umap-learn)
├──────────────────────────────────────────────────┤
│ 07a Leiden 聚类 (auto resolution)                │  scanpy + igraph C++
├──────────────────────────────────────────────────┤
│ 07b recall-validated 聚类 (knockoff + wilcoxon)  │  ⚡ scvalidate (Rust 22×)
├──────────────────────────────────────────────────┤
│ 08  scib metrics (iLISI/cLISI/graph_conn/kBET)   │  ⚡ scatlas.metrics.scib_score
└──────────────────────────────────────────────────┘
```

## 用法

v1 起 recall 为必备步骤,输出 `RecallComparisonReport` 到 `adata.uns`。

```python
from scatlas_pipeline import run_pipeline

adata = run_pipeline(
    "data/epithelia_full.h5ad",
    batch_key="orig.ident",
    integration="bbknn",        # "none" | "bbknn" | "harmony" | "all"
    leiden_resolutions=[0.05, 0.1, 0.2, 0.3, 0.5],
    leiden_target_n=(3, 10),    # major lineage level (epithelia/immune/stromal)
    recall_max_iterations=20,
    recall_fdr=0.05,
    recall_scratch_dir=None,    # None → tempfile 默认(≥30k 自动走 oom backend)
    out_h5ad="atlas.h5ad",
)
```

或 CLI:

```bash
python -m scatlas_pipeline.run --config configs/epithelia_157k.yaml
```

## 性能参考

157k × 16337 epithelia(WSL2 16-core):
| 阶段 | 时间 |
|---|---|
| load + QC | ~25s |
| lognorm | ~15s |
| PCA (auto) | ~80s |
| BBKNN + connectivities | ~26s |
| Harmony | ~15s |
| UMAP | ~5s |
| Leiden (auto resolution) | ~20s |
| scib metrics | ~3s |
| **总计** | **~3 min** |

recall 在 10k scale 200s(vs R 640s, vs Python 4461s);≥30k 时自动走 anndata-oom backend,157k 峰值内存 ~2.5-7 GB。

## 依赖

需要同一 virtualenv 里:
- `scatlas` (editable install from `../scatlas/crates/scatlas-py`)
- `scvalidate` + `scvalidate_rust` (from `../scvalidate_rewrite`)
- scanpy, anndata, igraph, leidenalg, scipy, numpy

# 下游分析路线图（v2-P13 起）

**目标**：在 `atlas_222k_v2p12.h5ad` 这种已经走完 QC → 集成 → 聚类的产物上，
继续做生物学解读 + 把热点步骤 Rust 化。

**起点数据**：`F:\down_ana\2026-04-26_atlas_222k_v2p12\data\atlas_222k_v2p12.h5ad`
（222k cells × 20055 genes，4 路集成 + CHAMP k=8 聚类完成，counts 在
`layers["counts"]`，3 级 GT 注释 ct.main / ct.sub / ct.sub.epi 可校验）

---

## 总体设计原则（沿用上游的成功经验）

1. **Python 主体 + Rust 内核**：算法层 Python，重计算 Rust 内核（pyo3 wheel）
2. **Cache-first**：所有产物写回 h5ad，下次免重算
3. **Validation-driven**：先用现成 reference impl（scanpy / decoupler-py / gseapy）跑一遍拿 ground truth，再做 Rust 加速 + 数值一致性 check
4. **每阶段独立可发**：阶段间 dict-of-arrays 接口，互不耦合
5. **微基准前置**：上游教训三条已沉淀（`feedback_rust_speedup_assumption` / `feedback_blas_bound_multiprocess` / `feedback_two_stage_knee_picker`）—— Rust 化前必先 microbench 真数据，不是 synthetic

---

## 阶段划分

### Stage 1 — DEG / Marker Genes（v2-P13）
**目标**：每个 cluster 找出显著高表达的基因（Mann-Whitney U test，one-vs-rest）

**工作量**：2-3 天（Python 1 天 + Rust 1-2 天）

**输入**：`adata.layers["counts"]` + `adata.obs["leiden_bbknn_champ"]`
**输出**：
- `adata.uns["rank_genes_champ"]` —— per-cluster top-k DataFrame (gene, log2FC, pval, padj, scores)
- `F:\down_ana\<date>_downstream\01_deg\` 下的 CSV + heatmap PNG

**Python 实现**：包装 `scanpy.tl.rank_genes_groups(method="wilcoxon")`
**Rust 加速目标**：复活 v1 删掉的 `wilcoxon.rs`，预期 50-100× vs scanpy（v1 是 100× 量级）
**验证**：与 scanpy 输出对照 |Δ pval| < 1e-6，rank order 一致

### Stage 2 — Cell-Type Annotation（v2-P14）
**目标**：把 leiden cluster 映射到细胞类型（基于 marker panels）

**工作量**：1-2 天（无 Rust，纯逻辑）

**输入**：Stage 1 的 DEG + 一个 marker panel（PanglaoDB / CellMarker / 自定义）
**输出**：
- `adata.obs["celltype_auto"]` —— per-cell 推断的类型
- 与 GT (ct.main / ct.sub) 的混淆矩阵 + 准确率
- `F:\down_ana\<date>_downstream\02_annotate\` 下注释表 + UMAP 着色图

**算法**：
- `marker_score`：对每个 cluster 算 panel-vs-DEG 的富集分（hypergeometric）
- 选 top score panel 作为 cluster 注释
- 可选：`celltypist` / `singleR` 风格的 cell-level 投票

### Stage 3 — Pathway / GSEA Enrichment（v2-P15）
**目标**：对每个 cluster 的 DEG 做 KEGG / Reactome / GO 富集

**工作量**：3-5 天（Python 1 天 + Rust 2-4 天）

**输入**：Stage 1 的 ranked DEG list + MSigDB 数据库
**输出**：
- `adata.uns["enrichment_<cluster>"]` —— per-cluster pathway × score 表
- `F:\down_ana\<date>_downstream\03_gsea\` 下富集结果 + bubble plot

**Python 实现**：
- ORA (over-representation) 用 `scipy.stats.hypergeom`
- GSEA permutation 用 `gseapy`（慢，~1-5 min per cluster × 1000 perms）
**Rust 加速**：GSEA 的 1000 次 random ranking 是 embarrassingly parallel + 计算密集，预期 30-50× via Rust + rayon

---

## 后续候选（v2-P16+，未启动）

| Stage | 内容 | 预期工作量 | Rust 加速空间 |
|---|---|---|---|
| 4 | **Cell-cell communication** (LIANA / CellPhoneDB) | 1-2 周 | 中（ligand-receptor 表达矩阵 inner join + score） |
| 5 | **Differential abundance** (Milo) | 1 周 | 高（NB GLM 在 thousands of neighborhoods） |
| 6 | **Trajectory / pseudotime** (PAGA / Slingshot) | 1-2 周 | 低（algorithm 复杂） |
| 7 | **Sub-clustering refinement** | 0.5 天 | 复用上游 CHAMP picker，无新代码 |

---

## 代码组织

```
fast_auto_scrna/
└── downstream/                         (新模块，独立于 upstream pipeline)
    ├── __init__.py
    ├── deg/
    │   ├── __init__.py
    │   ├── wilcoxon.py                 (Stage 1 算法层，Python)
    │   └── filter.py                   (FDR + log2FC 过滤)
    ├── annotate/
    │   ├── __init__.py
    │   ├── marker_score.py             (Stage 2 算法层)
    │   ├── transfer.py                 (label transfer fallback)
    │   └── panels/                     (marker panel JSON/YAML)
    ├── gsea/
    │   ├── __init__.py
    │   ├── ora.py                      (Stage 3 ORA)
    │   └── permutation.py              (Stage 3 GSEA permutation)
    └── plotting/
        ├── deg_heatmap.py
        ├── marker_dotplot.py
        └── enrichment_dotplot.py

rust/crates/kernels/src/downstream/     (新模块，未来 Rust 加速核)
├── mod.rs
├── wilcoxon.rs                         (Stage 1 内核 — 复活 v1 的实现)
├── ora.rs                              (Stage 3 ORA — hypergeometric tail)
└── gsea.rs                             (Stage 3 GSEA permutation + ES)
```

## 输出组织

```
F:\down_ana\
├── 2026-04-26_atlas_222k_v2p12\        (上游 CHAMP 交付，已有)
│   ├── METADATA.md
│   ├── data/, plots/, logs/
│
└── <date>_downstream\                  (按交付批次建目录)
    ├── 01_deg\
    │   ├── deg_per_cluster.csv
    │   ├── deg_top20.csv
    │   └── plots/  (heatmap, top markers)
    ├── 02_annotate\
    │   ├── cluster_to_celltype.csv
    │   ├── confusion_vs_gt.csv
    │   └── plots/  (UMAP colored by celltype, confusion heatmap)
    └── 03_gsea\
        ├── enrichment_per_cluster.csv
        └── plots/  (bubble plot, top pathways per cluster)
```

---

## 依赖追加（pyproject.toml）

Stage 1: 已有 scanpy
Stage 2: 可能加 `celltypist>=1.6`（可选，一开始不用）
Stage 3: 加 `gseapy>=1.1` + `decoupler-py>=1.6`（基准 impl，等 Rust 加速接好后可考虑去掉）

---

## 立即下一步

按 Stage 1 → 2 → 3 顺序开干。

**Stage 1 起步**：
1. `mkdir fast_auto_scrna/downstream/deg/` + `__init__.py`
2. 写 `wilcoxon.py` —— Python 层调 `scanpy.tl.rank_genes_groups`，规范输出 schema
3. 在 `atlas_222k_v2p12.h5ad` 上跑一遍，把 8 个 cluster 的 top markers 存到 `01_deg/deg_per_cluster.csv`
4. **stop and review** —— 看输出是否合理，再决定要不要立即 Rust 化
5. （如果合理）Rust：复活 v1 `wilcoxon.rs`，PyO3 wrap，`fast_auto_scrna/downstream/deg/wilcoxon.py` 加 Rust dispatch + sklearn fallback
6. 微基准 + 数值对照
7. 提交，进 Stage 2

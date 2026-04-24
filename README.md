# fast_auto_scRNA v2

端到端单细胞 RNA-seq 图谱分析管线 —— Rust 加速的核心内核 + Python 编排层，
**按管线阶段组织**（而不是按历史库划分）。每个阶段都是自包含的模块。

---

## 🟢 状态 (2026-04-24，分支 `main`)

**已完成**
- v2 工作树位于 `F:/fast_auto_scRNA_v2`，分支 `main`（由原 `v2` 分支重命名而来）
- 从 v1 提交 `c1107e8` 切出（包含 GS-2 graph-silhouette 管线接线 + 通过验证的 222k 图谱 smoke 测试）
- v1 遗留目录（`scatlas/`、`scatlas_pipeline/`、`scvalidate_rewrite/`、`scripts/`、`docs/images/`、`docs/superpowers/`）已清除 —— **166 文件、26 426 行**，提交 `32787b2`
- 新的阶段化骨架：`fast_auto_scrna/{io, preprocess, pca, integration, neighbors, scib_metrics, umap, cluster, rogue, common, plotting}` + `rust/crates/{kernels, py_bindings}`
- **V2-P1**（`91e2aef`）—— Rust 内核 + PyO3 绑定从 v1 `scatlas-core` / `scatlas-py` 迁移完成：
  - `rust/crates/kernels/src/`：7 模块 —— `bbknn`、`fuzzy`、`harmony/`、`metrics/`、`pca`、`rogue`、`umap`
  - `rust/crates/py_bindings/src/`：绑定按 v2 阶段名重新注册 —— `_native.{pca, bbknn, harmony, fuzzy, metrics, umap, rogue}`
  - 丢弃 recall 专用件：`wilcoxon.rs` + `knockoff.rs` + PyO3 包装
  - `cargo check --workspace` + `cargo clippy --workspace --all-targets`：全绿
- **V2-P2**（`b6797af`）—— v1 `pipeline.py`（1238 行）+ `silhouette.py`（263 行）+ `scvalidate.rogue_py/core.py`（506 行）按阶段拆成 `fast_auto_scrna/` 下 **19 个新 `.py` 文件**。`PipelineConfig` 去掉所有 recall 字段；`_run_recall_for_route` 整段删
- **V2-P3**（`3d91d28`）—— 端到端跑通：
  - `uv venv --python 3.10` + `maturin develop --release`（~90 s）+ `uv pip install -e .`
  - 新增 runtime 依赖：`scikit-misc`、`statsmodels`
  - `tests/test_smoke.py` 合成 smoke（500 cells × 500 genes × 2 batches × 3 groups）：**6.7 s 绿**，覆盖全部 Rust 内核 + sklearn fallback
- 测试数据：`data/pancreas_sub.rda`（3.5 MB）+ `data/StepF.All_Cells.h5ad`（1.66 GB，222 k cells × 20 055 genes，10 batches；3 级 ground-truth 标签 ct.main / ct.sub / ct.sub.epi）

**下一步（按顺序）**
1. **V2-P4** —— 文档收尾；`benchmarks/smoke_222k.py` 真数据首次 v2 复跑结果写回 ROADMAP 基线表。
2. **GS-3** —— 在 `rust/crates/kernels/src/silhouette.rs` 实现 Rust `silhouette_precomputed` 内核，把 222k silhouette 扫描从 890 秒压到 ~20 秒。派发钩子已在 `cluster/resolution.py::_silhouette_impl` 里埋好。
3. **SCCAF-Rust** —— Rust logistic-regression + CV 取代 sklearn LR。派发钩子已在 `scib_metrics/sccaf.py` 里埋好。
4. **度量审计** —— 222k BBKNN silhouette 曲线对 k 单调，需要判断是真信号（atlas 尺度的细粒度子簇）还是 graph-silhouette 方法本身的弱点（可考虑 modularity / 密度感知变体）。

按阶段的详细分解和性能基线见 [ROADMAP.md](ROADMAP.md)。

---

## 快速开始

```bash
# 1) 创建虚拟环境
cd F:/fast_auto_scRNA_v2
uv venv --python 3.10
source .venv/Scripts/activate      # Windows Git Bash
# 或：source .venv/bin/activate    # WSL / Linux

# 2) 编译 Rust wheel + editable 安装
maturin develop --release -m rust/crates/py_bindings/Cargo.toml
pip install -e .

# 3) 跑测试
pytest tests/ -v
```

完整安装步骤（Windows + WSL）见 [INSTALL.md](INSTALL.md)；进度 / 待办见
[ROADMAP.md](ROADMAP.md)。

## 目录布局 —— 按管线阶段组织

| # | 模块 | 职责 |
|---|------|------|
| 01 | `fast_auto_scrna/io/` | 加载 h5ad / rda / Seurat-qs；逐 cell QC 过滤 |
| 02 | `fast_auto_scrna/preprocess/normalize.py` | normalize_total + log1p |
| 03 | `fast_auto_scrna/preprocess/hvg.py` | 高变基因选择 |
| 04 | `fast_auto_scrna/preprocess/scale.py` | z-score + max 截断 |
| 05 | `fast_auto_scrna/pca/` | 随机化 PCA（Gavish-Donoho 自动 n_comps）—— **Rust** |
| 06 | `fast_auto_scrna/integration/` | BBKNN / Harmony 2 / none —— **Rust** |
| 07 | `fast_auto_scrna/neighbors/` | kNN + fuzzy_simplicial_set connectivities —— **Rust** |
| 08 | `fast_auto_scrna/scib_metrics/` | iLISI / cLISI / graph_conn / kBET / silhouette —— **Rust** |
| 09 | `fast_auto_scrna/umap/` | UMAP layout 优化 —— **Rust** |
| 10 | `fast_auto_scrna/cluster/` | Leiden + **graph-silhouette resolution 选择器**（本项目新方法）|
| 11 | `fast_auto_scrna/rogue/` | 按簇纯度（entropy + loess）—— **Rust** |
| — | `fast_auto_scrna/config.py` | `PipelineConfig` dataclass —— 所有参数集中一处 |
| — | `fast_auto_scrna/runner.py` | `run_from_config(cfg, adata_in=None)` —— 主入口 |
| — | `fast_auto_scrna/common/` | 共享的稀疏矩阵 / I/O 辅助工具 |
| — | `fast_auto_scrna/_native/` | 编译后 Rust 绑定的薄 re-export 层 |

## Rust workspace

```
rust/
├── Cargo.toml                         workspace 根
└── crates/
    ├── kernels/                       纯 Rust 算法内核（rlib，无 PyO3）
    │   └── src/
    │       ├── pca.rs
    │       ├── bbknn.rs
    │       ├── harmony/
    │       ├── umap.rs
    │       ├── fuzzy.rs               (fuzzy_simplicial_set)
    │       ├── metrics/               (lisi, graph_conn, kbet)
    │       ├── rogue.rs               (entropy_table + calculate_rogue)
    │       └── silhouette.rs          (graph silhouette —— 新方法，取代 recall)
    └── py_bindings/                   PyO3 → fast_auto_scrna._native
```

## v2 明确**不包含**的内容

- **`recall`** —— scvalidate 的 recall 簇数选择器整体放弃。簇数选择改由
  `cluster/resolution.py` 里的 graph-silhouette 优化器接管，后者在 Leiden
  使用的同一张 connectivity 图上评估，可扩展到全图谱尺度。
- **`wilcoxon` / `knockoff` Rust 内核** —— 只服务于 recall，一并丢弃。
- **`RecallComparisonReport`** —— 没有 baseline-vs-recall 报告，silhouette
  曲线本身就是诊断工具。

## 测试数据

- `data/pancreas_sub.rda` —— 1000 cell 胰腺谱系，1 batch
  （软链 → `F:/NMF_rewrite/pancreas_sub.rda`）。单元测试基准。
- `data/StepF.All_Cells.h5ad` —— 222 529 cells × 20 055 genes 前列腺图谱，
  10 batches，ct.main（3 类）/ ct.sub（7 类）/ ct.sub.epi（13 类）
  ground-truth 标签（软链 → `F:/NMF_rewrite/StepF.All_Cells.h5ad`）。
  图谱尺度基准。

## 历史沿革

v2 从 v1 提交 `c1107e8` 切出。v1 存活于
`F:/NMF_rewrite/fast_auto_scRNA_v1/`（分支 `v1`），现**已归档** ——
所有新工作都在这里进行。v1 保留完整的 recall / scvalidate 历史以供参考，
不做删除。

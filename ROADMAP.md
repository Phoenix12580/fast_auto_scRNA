# ROADMAP —— fast_auto_scRNA v2

## 状态 (2026-04-24，分支 `main`)

**v2 已从 v1 切出到单根、按阶段组织的工作区，端到端 smoke 已跑通。**
当前 HEAD：`3d91d28`（V2-P3 synthetic smoke pass）。起点：`c1107e8`
（v1 最新，含 GS-2 接线 + 222 k 图谱 smoke 通过）。

**V2-P0 done**（`b85c30b` + `32787b2` + 分支重命名）：
- 工作树 `F:/fast_auto_scRNA_v2`，分支 `main`（由 `v2` 重命名而来）
- 骨架：`fast_auto_scrna/` 11 个阶段子包 + `rust/crates/{kernels,py_bindings}` workspace + 空壳 `tests/benchmarks/docs/data`
- 文档：README / INSTALL / ROADMAP 按 v2 重写
- 遗产清理：`scatlas/`、`scatlas_pipeline/`、`scvalidate_rewrite/`、`scripts/`、`docs/superpowers/`、`docs/images/`、`setup.sh`、`UPDATE.md` —— 单次提交删掉 166 文件、26 426 行
- 测试数据软链：`data/pancreas_sub.rda` + `data/StepF.All_Cells.h5ad`

**V2-P1 done**（`91e2aef`）：
- `rust/crates/kernels/src/`：从 v1 `scatlas-core` 迁移 7 个模块（`bbknn`、
  `fuzzy`、`harmony/`、`metrics/`、`pca`、`rogue`、`umap`）；`rogue` 从
  v1 `stats/` 提升到顶层
- `rust/crates/py_bindings/src/`：PyO3 适配器从 v1 `scatlas-py` 过来，按
  v2 阶段名重新注册 —— `_native.{pca, bbknn, harmony, fuzzy, metrics,
  umap, rogue}`。BBKNN 从 v1 `ext/` 杂物包中独立为一级子模块
- 丢弃的 recall 专用件：`stats/wilcoxon.rs`（225 行）+
  `stats/knockoff.rs`（74 行）+ 它们的 PyO3 包装
- `kernels/Cargo.toml`：加 `libm = "0.2"`（`rogue::log1p` 要用）
- `cargo check --workspace` + `cargo clippy --workspace --all-targets`：全绿

**V2-P2 done**（`b6797af`）：
- v1 `scatlas_pipeline/pipeline.py`（1238 行） + `silhouette.py`（263 行）
  + `scvalidate.rogue_py/core.py`（506 行）全部按阶段拆入 `fast_auto_scrna/`
  下 19 个新 `.py` 文件：`config.py`、`runner.py`、`io/qc.py`、
  `preprocess/{normalize,hvg,scale}.py`、`pca/randomized.py`、
  `integration/{bbknn,harmony}.py`、`neighbors/knn_fuzzy.py`、
  `umap/layout.py`、`scib_metrics/{scib,sccaf}.py`、
  `cluster/{leiden,resolution}.py`、`rogue/{core,score}.py`、
  `plotting/comparison.py`
- `_native/` 目录脚手架删除 —— maturin 把编译产物直接放到
  `fast_auto_scrna/_native.{pyd,so}` 单文件模块
- `PipelineConfig` 去掉全部 recall 字段；`_run_recall_for_route` 整段删
- `from fast_auto_scrna import PipelineConfig, run_from_config` 在无编译
  扩展时也能 import（stage 模块统一用懒加载）

**V2-P3 done**（`3d91d28`）：
- `uv venv --python 3.10` + `maturin develop --release`（~90 s）+
  `uv pip install -e .` 全流程走通
- `pyproject.toml` 补 runtime 依赖：`scikit-misc`（seurat_v3 HVG loess +
  ROGUE entropy fit）、`statsmodels`（ROGUE FDR）
- `tests/test_smoke.py`：500 cells × 500 genes × 2 batches × 3 groups
  合成数据的 BBKNN 路径 end-to-end smoke，6.7 s 绿，覆盖全部 Rust 内核 +
  sklearn fallback（silhouettes / SCCAF / graph-silhouette 选择器）
- `.gitignore` 加 `*.pdb`（Windows maturin 产生的 debug 符号）

## 阶段清单

| # | 阶段 | 代码 | Rust？ | 状态 |
|---|------|------|--------|------|
| 01 | `io/` | 加载 + qc + 格式转换 | — | 从 v1 搬 |
| 02 | `preprocess/normalize` | normalize + log1p | 否（scanpy） | 已搬 |
| 03 | `preprocess/hvg` | HVG seurat/seurat_v3 | 否（scanpy） | 已搬 |
| 04 | `preprocess/scale` | z-score + max 截断 | 否（scanpy） | 已搬 |
| 05 | `pca/` | 随机 SVD + Gavish-Donoho 自动 | ✅ | 已搬 |
| 06 | `integration/bbknn` | HNSW + fuzzy | ✅ | 已搬 |
| 06 | `integration/harmony` | Harmony 2（较 R 提速 6×） | ✅ | 已搬 |
| 06 | `integration/none` | 纯 kNN | — | 已搬 |
| 07 | `neighbors/` | kNN + fuzzy_simplicial_set | ✅ | 已搬 |
| 08 | `scib_metrics/` | iLISI/cLISI/graph_conn/kBET/silhouette | ✅ | 已搬 |
| 09 | `umap/` | UMAP SGD layout | ✅ | 已搬 |
| 10 | `cluster/leiden` | scanpy Leiden 包装 | — | 已搬 |
| 10 | `cluster/resolution` | **graph-silhouette 选择器（新方法）** | ⚠️ sklearn fallback | **先搬 Python 侧，后 Rust-ify** |
| 11 | `rogue/` | entropy_table + calculate_rogue | ✅ | 已搬 |

## 近期待办

1. **V2-P4**：文档收尾（本文件 + README + INSTALL）、`benchmarks/smoke_222k.py`
   结果写回性能基线表。
2. **V2-P5**：更新 memory；把 v1 / scatlas / scvalidate_rewrite 标记为
   DEPRECATED（不删）；清理 v1 worktree（`F:/NMF_rewrite/fast_auto_scRNA_v1`）。

## 重组完成后的主线工作

1. **GS-3**：Rust `silhouette_precomputed` 内核 —— smoke 里的 889.9 s sklearn
   fallback 压到 ~20 s。落在 `rust/crates/kernels/src/silhouette.rs`。
   `cluster/resolution.py::_silhouette_impl` 已预留派发钩子，内核就绪后是
   一行切换。
2. **SCCAF-Rust**：Rust logistic-regression + CV accuracy 内核，取代
   `scib_metrics/sccaf.py` 里的 sklearn LR。派发钩子已埋在
   `sccaf.py`：一旦 `_native.sccaf.lr_cv_accuracy` 可用即自动切换。
   候选实现：`linfa-logistic` + 自写 k-fold（小作坊），或手撸 L-BFGS（透明）。
3. **222k smoke 复跑**（用 Rust silhouette + SCCAF 内核）—— 预期总 wall 约 4
   min（现 18 min）。
4. **Harmony2 silhouette smoke** —— 在 222k 上对比 Harmony 图 vs BBKNN 图
   的 silhouette 曲线。
5. **graph-silhouette 度量审计** —— 222k BBKNN smoke 的曲线对 k 单调，
   需判断是真信号（簇数越多图越贴合）还是度量本身的弱点。考虑 modularity
   或密度感知变体。
6. **OOM-1**：接入 `anndataoom` 的分块 preprocess（normalize / log1p /
   HVG / scale）—— 当前 scanpy 全内存，上不去更大的图谱。
7. **rda → AnnData loader**：目前 `data/pancreas_sub.rda` 是 Seurat 对象，
   `rdata` 不会转。补一个 Seurat → AnnData 转换器（或直接用 h5Seurat 路径）
   后就可以把 pancreas 纳入 pytest。

## 明确丢弃（来自 v1）

| 丢弃项 | 原因 |
|--------|------|
| `scvalidate_rewrite/scvalidate/recall_py/*` | recall 由 graph-silhouette 取代 |
| `wilcoxon` Rust 内核 | 只服务于 recall |
| `knockoff` Rust 内核 | 只服务于 recall |
| `recall_oom_backend` | recall 专用 |
| `RecallComparisonReport` | recall 专用 |
| `run_recall` 配置字段 + 调用方 | recall 专用 |
| `test_recall_*` 测试 | recall 专用 |

## 性能基线（222k StepF 图谱，BBKNN 路径）

**以下数据来自 v1（c1107e8）smoke**。v2 在同一数据、同一配置下的 smoke
脚本已写好（`benchmarks/smoke_222k.py`），首次 v2 复跑结果待回填。

| 阶段 | Wall (v1) | 备注 |
|------|-----------|------|
| load + qc | 19.5 s | anndata eager 加载 |
| lognorm | 12.2 s | scanpy |
| hvg (seurat) | 24.6 s | scanpy；seurat_v3 在 222k×10 batches 会段错误，故用 seurat flavor |
| scale | 10.9 s | scanpy |
| **pca**（auto, Rust） | **82.8 s** | GD → 13 comps |
| **bbknn + fuzzy**（Rust + HNSW） | **35.1 s** | kNN 6.7M 边 → fuzzy CSR 12.3M nnz |
| metrics | 1.4 s | scIB（v1 smoke 无 ground-truth 标签；v2 使用 `ct.main` 作 GT） |
| **管线小计** | **187.5 s (3.1 min)** | |
| silhouette 优化器 × 16 res × 50 iter | 889.9 s | **sklearn fallback —— Rust 化即 GS-3** |
| **smoke 总 wall** | ~18 min | |

222k 的 graph-silhouette 曲线对 k 单调；k ∈ [3, 10] 内最优为
res=0.20 / k=10，silhouette = 0.00033 ± 0.00009。需做度量审计。

### v2 复跑（运行 `python benchmarks/smoke_222k.py` 获得）

结果待回填。

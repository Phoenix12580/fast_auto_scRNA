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

1. ~~**GS-3**：Rust `silhouette_precomputed` 内核~~ **（已完成，5ba6d83）**
   —— 单次调用比 sklearn 快 **40.5×**（13.05 ms → 0.32 ms on 1000×1000
   f32），数值差 **2.98e-8**。合成 + 222k 双层 consistency 验证通过。
   atlas-scale 实际收益有限（~1% of total wall）因为 Leiden 本身占大头；
   真正节省来自配套的 **stratify dedup**（Phase-2 Leiden −17-21%，总 wall
   23.0 min → **19.6 min**，全局 **−15%**）。
2. ~~**SCCAF-Rust**：Rust logistic-regression + CV accuracy 内核~~
   **（已评估并否决，2026-04-24）**：用 linfa-logistic + 自写 stratified
   k-fold + z-score 预处理 + L-BFGS 多 alpha 重试梯做过一版，数值上
   与 sklearn 一致（|Δ|<0.005 on 合成 well-separated 数据，11/11 测试过），
   但**速度比 sklearn 慢 ~1.8×**（10k×15×10-class 3-fold CV：Rust 727 ms
   vs sklearn 404 ms）—— 因 linfa 的 argmin L-BFGS 是纯 Rust 没用 LAPACK，
   而 sklearn 走 scipy 的 Fortran lbfgs。SCCAF 在 222k 各 route 只占 13 s
   (BBKNN) / 42 s (none)，非瓶颈，Rust 化是净回归。保留 `scib_metrics/
   sccaf.py` 为纯 sklearn 实现。经验写入 memory。
3. ~~**Rust Leiden 内核**~~ **（2026-04-24 改走 Python-MP，已完成 GS-4）** ——
   微基准揭示 Leiden 的 C++ igraph 单次成本不可压（48.78s/res on 222k
   bbknn），瓶颈是 **串行扫 5 个 resolution**。把 sweep 改为
   `ProcessPoolExecutor` 后（pickle 图 99MB / 5 workers initializer 一次性
   分发）：244.80s → 66.01s，**3.71× 加速**；`optimize_resolution_graph_
   silhouette` 端到端 ~250s → **70.86s (3.5×)**。leidenalg 同 seed 确定，
   parallel == sequential **bit-identical**（3 pytest pass）。Rust 版被否
   决的理由：理论上限 5× 仅比 MP 多 1.3×，复现 leidenalg refinement/
   aggregation 语义 + ARI parity 测试估 1-2 周工作，收益风险比劣。
   networkit 跳板也无必要。经验：memory/feedback_rust_speedup_assumption
   已覆盖该模式（sklearn/scipy 包 Fortran 的东西别 Rust 化）。
4. **Harmony2 silhouette smoke** —— 在 222k 上对比 Harmony 图 vs BBKNN 图
   的 silhouette 曲线。（baseline 跑完已经有数据：BBKNN iLISI=1.00
   vs Harmony theta=4 iLISI=0.11 —— 明确 BBKNN 在该图谱上完胜，Harmony
   需调参或换模式。）
5. ~~**graph-silhouette 度量审计**~~ **（2026-04-24 完成，换成 conductance）**
   —— 诊断发现 1000-cell 子采样在 222k kNN 图上只剩 0.012% 连边，
   距离矩阵 99.89% 是 1，silhouette 纯在噪声底漂移，单调递增只是
   "簇越多越有机会撞上 1-2 条边"的赝信号。ARI vs ct.main 曲线和旧度量
   **完全反向**（r=0.05 ARI=0.69 → r=0.50 ARI=0.20，旧度量选后者，
   `leiden_target_n=(3,10)` 的 clip 是运气救场）。
   对比候选（222k BBKNN）：
   - 旧 graph_silhouette: r=0.50（错）
   - conductance: **r=0.05（ARI=0.69 ✓）** — 全图 O(nnz) 无子采样，确定
   - embedding silhouette: r=0.05 ✓ — 正确但慢 3×
   - modularity: r=0.50 ✗ — 稠密图上单调，废
   新默认 `cfg.resolution_optimizer="conductance"`，旧路径保留为向后兼容。
   3 个新 pytest（perfect split / worst split / 2-blob 合成）过，整套 12/12 过。
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

### v2 复跑结果（`python benchmarks/smoke_222k.py`，2026-04-24）

**BBKNN-only 路径：9.3 min wall（v1: 18 min，快 48%）。** 三个 sklearn
silhouette 在 222 k 要 ~45 min，所以完全移除（V2-P5）。

**all-mode 路径（3 种 integration 同跑）：19.6 min wall（post GS-3）** vs
初版 23.0 min（−15%）。见下方 all-mode 对比表。

| 阶段 | v2 Wall | v1 Wall | Δ |
|------|---------|---------|------|
| load + qc | 36.0 s | 19.5 s | 慢（可能冷盘）|
| lognorm | 7.5 s | 12.2 s | ✅ −39% |
| hvg (seurat) | 20.8 s | 24.6 s | ✅ −15% |
| scale | 6.2 s | 10.9 s | ✅ −43% |
| **pca (Rust, GD → 15 comps)** | **53.6 s** | 82.8 s | ✅ **−35%** |
| **bbknn + fuzzy (Rust + HNSW)** | **21.7 s** | 35.1 s | ✅ **−38%** |
| umap (Rust) | 11.8 s | — | |
| scIB metrics (Rust, 4 个) | 1.4 s | 1.4 s | = |
| Leiden + graph-silhouette 扫描 | 296.8 s | 889.9 s | ✅ −67%（n_iter 100→50 + 5 res）|
| ROGUE (Rust + LOESS) | 50.1 s | — | |
| SCCAF (sklearn LR CV) | 15.9 s | — | |
| **TOTAL** | **533 s (8.9 min)** | ~18 min | ✅ |

scIB 评分（BBKNN 路径，`ct.main` 3-class GT）：

| 指标 | 值 | 注 |
|------|------|------|
| iLISI | 1.000 | batch 混合极好 |
| cLISI | 0.960 | cell-type 保留良好 |
| graph_connectivity | 1.000 | per-label 最大 CC 几乎覆盖全部 |
| kBET acceptance | **n/a** | kBET 在 BBKNN 的 batch-balanced kNN 上按构造必定 ≈ 0 —— `kbet()` wrapper 现会检测恒定 batch 分布并返回 NaN + note（见下）|
| ROGUE mean | 0.790 | 10 cluster，中等纯度（3 纯 / 4 中 / 3 杂）|
| SCCAF | 0.983 | cluster 在 embedding 上高度线性可分 |
| **scIB mean** | **0.933** | iLISI / cLISI / graph_conn / ROGUE / SCCAF 的 nanmean；kBET 合理被跳过 |
| **Overall (heatmap)** | **0.97** | 0.35·Batch(1.00) + 0.45·Bio(0.98) + 0.20·Homo(0.89) / 总权重 |

Leiden 选到 **k=10 @ res=0.20**。silhouette 曲线对 k 单调（r=0.05 s=0.00025
→ r=0.50 s=0.00050），与 v1 观察一致 —— **metric audit 待办确认仍在 v2 成立**。

**图输出**（`benchmarks/out/smoke_222k_plots/`）：
- `umap_bbknn.png` —— 三图并列（batch / ct.main / leiden）
- `silhouette_curve_bbknn.png` —— 曲线 + 选点标记
- `rogue_per_cluster_bbknn.png` —— 10 个 cluster purity bar（三色分档）
- `scib_summary_bbknn.png` —— 单行 scIB 指标 heatmap

### all-mode 222k 对比（2026-04-24，3 方法同跑）

命令：`python benchmarks/smoke_222k.py --integration all`（默认）
输出：15 张图 + 一个 AnnData h5ad（含 `uns["scib_comparison"]` DataFrame）

**post-GS-5 baseline (conductance optimizer + parallel Leiden sweep)**：

| 指标 | none | bbknn | harmony |
|------|------|-------|---------|
| iLISI | 0.025 | **1.000** | 0.114 |
| cLISI | 0.997 | 0.957 | 0.996 |
| graph_connectivity | 0.999 | 1.000 | 0.999 |
| kBET | 0.000 | n/a（batch-balanced）| 0.008 |
| ROGUE mean | 0.780 | 0.776 | 0.769 |
| SCCAF | 0.990 | 0.992 | 0.996 |
| **scIB mean** | 0.505 | **0.986** | 0.529 |
| picked res / k | r=0.05 / k=14 | r=0.05 / k=8 | r=0.05 / k=8 |

**Wall time**: **12.0 min**（post GS-3 baseline 19.6 min → **−39%**，来自 GS-4
并行 Leiden 把 Phase-2 per-route 从 ~220s 砍到 ~125s）。

**结论（此 222k 前列腺图谱 specifically）**：**BBKNN 胜出**（Overall 0.99
vs none/harmony 0.51/0.53）。Conductance optimizer 在三路由上都选中
r=0.05：BBKNN/Harmony k=8（吻合 "3 大谱系 × 2-3 亚型" 结构），none k=14
（未整合数据碎片更多，符合预期）。旧 graph_silhouette 在同数据上选
r=0.20-0.50 / k=10-34，对 ct.main 的 ARI 低至 0.20；新度量纠正为
k=8 且 ARI ~0.69。Harmony `theta=4` 仍未解决（iLISI 0.114）—— 独立 tuning
任务，不是管线问题。

### GS-3 与 baseline 的 consistency

**Rust 确定性内核完全匹配**：iLISI / graph_connectivity / kBET 三个路径
全部位相同；picked resolution 三路径均一致（r=0.50 / r=0.20 / r=0.10）；
scIB mean 差异 ≤ 0.0007。

**Label-dependent 指标有 <0.005 微漂**：cLISI（经由 ct.main 标签 neighbor
gather）和 ROGUE / SCCAF（经由 Leiden 标签）。根因是 HNSW 并行构图的
浮点非确定性 + 去掉 stratify warmup 后 igraph Leiden 的 RNG 序列改变。
定性结论（BBKNN 最好 / picked k）稳定。

### 已解决 / 已解释

1. **kBET = 0.000 根因已定位（2026-04-24）**：BBKNN 的 batch-balanced kNN 让
   每个 cell 的邻居 batch 分布恒为 `[3, 3, ..., 3]`（per-batch 均匀），与
   倾斜的全局 batch 分布做 chi2 时观察 vs 期望差距极大，每个 cell 都被
   拒绝。这不是 bug 是 BBKNN 的构造属性，scib-metrics 官方文档有提及
   "kBET underperforms on batch-balanced methods"。**修复**：
   `scib_metrics.kbet()` 现会采样 100 个 cell 检测 batch 分布恒定性，若恒定
   则返回 `acceptance_rate=NaN` + note。Harmony / none 路径走普通 kNN
   （Harmony on X_pca_harmony, none on X_pca），kBET 正常计算 —— 合成 smoke
   验证 Harmony kBET = 0.958，none kBET = 0.108，符合预期。

2. **三个 sklearn silhouette 完全移除（2026-04-24）**：`label_silhouette`、
   `batch_silhouette`、`isolated_label_silhouette` 都是 O(N²)，222k 单路径
   要跑 ~45 min，对 atlas 尺度没有实用价值。删掉 `scib_score` 的
   `embedding` 参数 + 删掉 `compute_silhouette` config 字段 + 删掉
   plotting 的三个列。scIB 里批次维度只看 iLISI + kBET，生物维度看 cLISI
   + graph_connectivity，同样可诊断。

3. **plot_dir 成为唯一绘图控件（2026-04-24）**：去掉 `write_comparison_plot`
   的单文件路径模式。设 `plot_dir` 一个目录就全出：
   `integration_comparison.png`（大对比 UMAP）、`scib_heatmap.png`
   （方法 × 指标）、`rogue_comparison.png`、以及每个方法独立的 `umap_<m>`
   / `scib_summary_<m>` / `silhouette_curve_<m>` / `rogue_per_cluster_<m>`。
   `benchmarks/smoke_222k.py` 默认 `integration='all'`，对新图谱首次跑就
   产生完整的方法对比图，据此选最合适的方法。

### 待调查

1. **graph-silhouette 对 k 单调** —— v2 复现（r=0.05 s=0.00025 → r=0.50
   s=0.00050），确认不是 v1 的偶发现象。动作：实现 modularity / 密度感知
   变体对比。

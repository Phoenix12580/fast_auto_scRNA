# ROADMAP —— fast_auto_scRNA v2

## 状态 (2026-04-25，分支 `main`)

**v2 已从 v1 切出到单根、按阶段组织的工作区，端到端 smoke 已跑通。**
当前 HEAD：`ef53532`（v2-P9.1：CPU 礼让 + worker BELOW_NORMAL 优先级）。
起点：`c1107e8`（v1 最新，含 GS-2 接线 + 222 k 图谱 smoke 通过）。

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

## 当前 Sprint (2026-04-25)

按"快收益 → 大主线"排，今日预计 1 → 2 → 4，3 单独起天：

1. **[修] 自动选 → 人类决策停顿**（半小时）—— 多路由模式
   （`integration='all'` 且 `cfg.cluster_method is None`）跑完 Phase 2a +
   `scib_heatmap_pre_cluster.png` 后**早退**，打印 auto-pick 推荐 + 提示
   "重跑时设 `cfg.cluster_method=<route>` 进 Phase 2b"。单路由不变。
   动机：当前 Phase 2b 的 winner 是机器自动 argmax(scIB mean) 选的，没给
   人类看 heatmap 后改主意的窗口；scIB mean 在不同图谱上权重应不同，
   半自动 > 全自动。
2. ~~**[诊] ASW = 0 根因调查**~~ **（2026-04-25 完成）**—— 写
   `benchmarks/diagnose_asw.py`（h5py 直读 obs+obsm 绕开 anndata uns 兼容
   问题），在 `benchmarks/out/smoke_222k_all_v2p9.h5ad` 上分层抽样 200/类
   跑 sklearn 三 silhouette。**结论：原"ASW = 0"是误记**，三种都不是：

   | route   | label_asw | raw mean | batch_asw | iso_asw |
   |---------|-----------|----------|-----------|---------|
   | none    | 0.588     | +0.177   | 0.922     | 0.645   |
   | bbknn   | 0.588     | +0.177   | 0.922     | 0.645   |
   | harmony | 0.617     | +0.234   | 0.754     | 0.682   |

   - `X_pca_bbknn` ≡ `X_pca`（BBKNN 改图不改 embedding，见
     `_phase1_integration_umap` line 290-291），**所以 none/bbknn 三个
     embedding-level ASW 数值完全相同 —— BBKNN 路径用 embedding ASW
     评 batch 整合是 no-op，必须用图级 iLISI / kBET**
   - per-label: Epithelia 分离好（raw +0.30 ~ +0.47, 100% pos），
     Immune 中等（+0.16, 93% pos），Stromal 最差（+0.06, 仅 73-77% pos）
   - Harmony 比纯 PCA 提升 cell-type ASW（+0.06）但同时降低 batch ASW
     （-0.17），与 iLISI 的"Harmony 0.114 vs BBKNN 1.000"方向相反 ——
     **iLISI 是 kNN 局部度量，batch ASW 是类内全局几何，两者捕获不同**
   - 早期"ASW = 0"很可能是把 graph-silhouette resolution optimizer 的
     曲线值（v1 ≈ 0.00025-0.00050）当成了 embedding ASW

   **影响 #4**：Harmony θ 扫描值得跑（已在 ROADMAP），但不要期望 iLISI
   翻转 BBKNN 在该 atlas 的优势 —— BBKNN 的 batch-balanced kNN 在 iLISI
   上是构造性最优。诊断图 `benchmarks/out/diagnose_asw.png`。
3. **[主] OOM-1**（1-2 天）—— 接 `anndataoom` 分块 preprocess
   （normalize / log1p / HVG / scale），目标突破 scanpy 全内存上限。
   独立大坑，今日不开。
4. ~~**[小] Harmony θ 调参**~~ **（2026-04-25 决策跳过）**—— 用户确认
   按 Harmony2 教程默认参数，不调 θ。原因：sprint #2 显示 Harmony @ θ=4
   已经"label_asw 最高 + iLISI 最差"的过度矫正征兆方向（θ↑ 大概率拖
   bio 指标）；且 BBKNN 在该 atlas 上 iLISI 是构造性最优。如果以后换
   batch 平衡更好的图谱，再开 θ 调参。

5. **[新] ASW 三件套接回 + 默认开启**（2026-04-25 完成）—— 论文
   *Gao et al. Cancer Cell 2024*（张泽民组 cross-tissue fibroblast atlas）
   方法明确用 7 个 scib-metrics 指标 + ROGUE + SCCAF；V2-P5 砍掉的 3 个
   silhouette（label / batch / isolated）正是论文 Bio + Batch 维度的核心。
   重新接入：
   - `scib-metrics>=0.5` 加进 `pyproject.toml`（带 jax/jaxlib，CPU 模式
     无 GPU 依赖坑）。第一次 JAX JIT warmup ~5-15s，后续 ~25ms。
   - `fast_auto_scrna/scib_metrics/scib.py` 加 `label_silhouette` /
     `batch_silhouette` / `isolated_label_silhouette` 三个 wrapper
     （都走 scib-metrics JAX-jit chunked）；`scib_score()` 接受
     `embedding=None` 参数。
   - `cfg.compute_silhouette: bool = True`（默认开）；`runner.py` 把
     `embed_for_scib` 透传到 `_compute_scib_for_route`。
   - `plotting/comparison.py`：`SCIB_BATCH_METRICS` 加 `batch_silhouette`，
     `SCIB_BIO_METRICS` 加 `label_silhouette` + `isolated_label`，`Overall`
     heatmap 现在 9 列 + 4 摘要列，与论文 panel 一致。
   - 222k 全量 wall：单路由 ~5.5 min（label 130s + batch 55s + iso 140s），
     all-mode 16-17 min；管线总 wall 12 min → ~28 min。可设
     `compute_silhouette=False` 退回 12 min 模式。
   - 测试：19/19 pass，含新 `test_pipeline_compute_silhouette_off`
     验证 opt-out 路径。
   - 222k 数值（来自 sprint #2 全量 scib 结果）：harmony **label_asw
     0.6394** > bbknn/none 0.6059；harmony batch_asw 0.8346 > bbknn
     0.8140；iso 三路由 ~0.534。

### 之后（旧 V2-P 收尾）

- **V2-P4**：文档收尾（README + INSTALL）、`benchmarks/smoke_222k.py`
  结果写回性能基线表（部分已写）。
- **V2-P5**：更新 memory；把 v1 / scatlas / scvalidate_rewrite 标记为
  DEPRECATED（不删）；清理 v1 worktree（`F:/NMF_rewrite/fast_auto_scRNA_v1`）。

## INTEGRATION_METHODS 重组（2026-04-25）

按论文 panel 对齐 + 用户决策（"不算 none，加 scVI 和 fastMNN"）：

| route | 原 | 新 | 说明 |
|-------|:--:|:--:|------|
| `none` | ✓ | ✗ | **删除** —— atlas 上"未整合"基线无信息量 |
| `bbknn` | ✓ | ✓ | 保留，graph-level baseline |
| `harmony` | ✓ | ✓ | 保留，embedding-level，按 Harmony2 教程默认参数（不调 θ）|
| `fastmnn` | — | ✓ | **新增**（2026-04-25 done）—— Haghverdi 2018 / batchelor::fastMNN port |
| `scvi` | — | ✓ | **新增**（2026-04-25 done）—— scvi-tools VAE，n_latent=30，max_epochs=200 |

### fastMNN 实现（2026-04-25 done）

- **mnnpy 装不上**：0.1.9.5 的 Cython `_utils` 用 GCC-only 编译标志
  （`-ffast-math` / `-march=native` / `-fopenmp`），MSVC 不识别；维护
  停滞 2-3 年，无 Windows 友好 fork。**决策：自己写纯 Python 版**
  （`fast_auto_scrna/integration/fastmnn.py`，~200 行），用 hnswlib
  做跨批 kNN + numpy 做 correction smoothing。
- **算法**：cosine-normalize → 按 batch size 降序 → 顺序合并：每个非
  reference batch 找 MNN pairs → correction = `ref_pos - b_pos` →
  Gaussian kernel smoothing（bandwidth = median MNN-pair cosine
  distance × sigma_scale）→ 应用到该 batch。
- **222k microbench**：**3.4 min wall**（9 merges, 31k-94k pairs each,
  no skipped batches）。线性扩展估计 440k ~7 min。**Rust 化不需要**
  按 `feedback_rust_speedup_assumption` 教训。
- 测试：`test_pipeline_fastmnn_end_to_end`（500-cell 合成，端到端跑通
  + scIB 全部 finite + Leiden ≥ 2 簇）pass。

### scVI 实现（2026-04-25 done）

- 依赖：`scvi-tools>=1.1`（带 PyTorch 2.11 + lightning，~3 GB 磁盘）；
  Windows 默认装 CPU-only torch（`torch.cuda.is_available() == False`）。
- 实现：`fast_auto_scrna/integration/scvi_route.py`（薄包装），
  `setup_anndata` 用 `layer="counts"`（raw counts 来自 stage 02 写入的
  `adata.layers["counts"]`）+ `batch_key="_batch"`；默认只在 HVG 上训
  （`scvi_use_hvg=True` 走 `adata.var["highly_variable"]` 子集），
  atlas 上 ~2k HVGs vs 全 20k genes 训练 ~10× 加速。
- 默认参数（论文 Gao et al + scvi-tools 标准）：`n_latent=30,
  n_hidden=128, n_layers=1, max_epochs=200, gene_likelihood="zinb",
  dispersion="gene", accelerator="auto", batch_size=128, seed=0`。
- **222k CPU wall 估计**：全 200 epoch ~30-60 min（实测待跑）；
  GPU（CUDA）~2-5 min。
- 输出：`adata.obsm["X_scvi"]`（n_latent=30 dim）+
  `adata.obsm["X_pca_scvi"]`（同一份，与其他 route 命名一致）+
  `adata.uns["scvi"]`（diagnostic info dict，含 `actual_epochs` /
  `early_stopping` 等）。
- **CUDA 启用** + auto-epoch + early stopping（2026-04-25 优化）：
  - 默认装的是 CPU-only torch (`torch==2.11.0+cpu`)，重装为
    `torch==2.5.1+cu121` 后 `cuda available=True`（GTX 1660 SUPER, 6.4 GB）。
  - 第一次 GPU bench：`max_epochs=200` 估计 wall 107 min，但**模型 epoch
    ~15 就收敛**（loss 581→507→507），200 是 5× overshoot。
  - 改默认：`scvi_max_epochs: int | None = None`（走 scvi-tools 启发式
    `min(round((20000/n_cells)*400), 400)` —— 222k 36 epoch / 444k 18 epoch /
    小数据 400 epoch）+ `scvi_early_stopping=True`。
  - **222k 实测（auto-epoch）：20.4 min wall，36 epochs**。比 200-epoch
    估计快 5.2×，且无需手调 epoch。
  - 1660 SUPER 利用率 14-36% —— 瓶颈在 CPU/IO（小 batch_size + 0
    dataloader workers），不在 GPU 算力。未来优化项（不必现在做）：
    `precision="16-mixed"`、`batch_size=256`、`dataloader_kwargs.num_workers=4`。
- 测试：
  - `test_pipeline_scvi_end_to_end`（500-cell 合成，`max_epochs=5` 走
    完整 dataflow，60s wall）pass
  - 4-route gate test 默认开 scvi `max_epochs=5`，避免 CI 跑 30 min
- 论文一致性：`scvi-tools v1.3.3`（论文用的 v1.1.2 是同一 SCVI 模型，
  仅 lightning + 训练 API 微调）。

## 测试套件（2026-04-25 终态）

- **21 tests pass / 3.6 min wall**（之前 18 tests / 1.7 min）。
- 增量：fastmnn smoke + scvi smoke + multi-route 4-routes 校验。
- scvi 测试 wall ~60s（torch init + 5 epoch CPU 是主要成本）。

## v2-P11 实测 + ASW 加速方案修订（2026-04-26）

### 已落地：Phase 2a multi-process scaffolding（默认关，commit 641708f）

`runner._phase2a_scib_all_routes` + `cfg.scib_parallel`。代码正确、数值
bit-identical（4 routes × 7 metrics 全部 \|Δ\| ≤ 1.79e-7 vs v2-P10
baseline）。**默认关闭**，因为：

| | Sequential | Parallel (4w × 4 thread, 16-core WSL) |
|---|---:|---:|
| 4-route Phase 2a wall | **1640.9 s (27.3 min)** | **1787.7 s (29.8 min)** |
| 加速比 | — | **0.92× — 反而变慢** |
| 单 route 在 par 下 | — | ~4× 慢（bbknn 427 → 1710 s） |

scib-metrics 的 JAX silhouette 是 **BLAS-saturated**：单进程已经吃满
16 线程，4-way 切分等于把每个 route 的 BLAS 算力降到 1/4，总 wall 不变
甚至略亏。Synthetic 500-cell 测的 2.34× 是误导，因为小矩阵 BLAS 不主导。

详见 commit message + `feedback_blas_bound_multiprocess.md` 记忆。

### 修订 ASW 加速方向：GPU silhouette（推荐）

下方原 "Rust + BLAS" 方案的 22× 估计**已被 v2-P11 证据推翻**：scib-metrics
本身已经走 XLA→BLAS sgemm，Rust+BLAS 在同一 CPU 上**不会有数量级提升**，
最多 1.5-2×（边缘 chunking / Python overhead）。真正能拿数量级的只有 GPU。

| 方案 | 期望 | 数值一致性 | 工作量 | 风险 |
|------|---:|---|---:|---|
| **A. torch.cuda silhouette** | 460 s → ~20 s/route，**~25×** | FP32 sgemm reduction order 内一致（\|Δ\| < 1e-4） | 1 天 | WSL 装 CUDA torch（cudnn 600 MB 之前下载失败过，要重试） |
| B. Rust + ndarray-linalg | 460 s → ~250 s/route，~2× | 同 | 1-2 周 | Windows openblas-static 配置；收益不抵成本 |
| C. 子采样（stratified n=50k） | 460 s → ~25 s/route，~18× | 数值漂移（不一致） | 0.5 天 | 违反 "结果一致" 硬约束 |
| D. kNN-graph silhouette（替算法） | 460 s → < 1 s/route，> 100× | 算法不同（不一致） | 1 天 | 同 C |

**推荐 A**。下次会话先验证 WSL CUDA torch 装通（重试 cudnn 下载），
然后写 `fast_auto_scrna/scib_metrics/silhouette_torch.py`：

```python
# 三函数：label / batch / isolated
# - X: (n, d) float32 → torch.from_numpy(X).cuda()
# - 分块 pairwise dist: 每次 chunk × n，BLAS 走 cuBLAS sgemm
# - per-cell (a, b) 累加用 torch.scatter_add
# - 与 scib-metrics 默认参数一致：euclidean, rescale=True
```

接进 `scib.py` 用 try/fallback：先试 torch.cuda，失败回落 scib-metrics。

### 验证策略
复用本次的 `benchmarks/validate_scib_parallel_222k.py` 框架：把里面
`scib_parallel=True` 那次替换为 `use_torch=True`，对比同一份 v2-P10 h5ad
的存储 baseline。期望：\|Δ\| < 1e-4，wall ~80 s vs 1640 s 的 21× 加速。

### 跳过条件
当前 ASW 27 min 在 100 min 总流水线里占 27% — 不算最大瓶颈了。如果用户
觉得「够用」（如 2026-04-26 决定），就此打住，下个加速目标转向 Phase 2b
的 Leiden 150-pt sweep（28 min）或 SCCAF。

---

## 待办：ASW Rust 化（2026-04-25 立项，**已被 v2-P11 evidence 推翻 — 见上**）

**动机**：当前 222k all-mode wall ~55 min，其中 ASW × 4 routes 占 22 min
（**40%**），是除 scvi（PyTorch，不可 Rust 化）外最大开销。scib-metrics
0.5.1 已经是 JAX-jit chunked SOTA Python 实现，但 Rust + BLAS gemm 还能
再快 **数十倍**（理论 < 1 min for 4 routes）。用户 2026-04-25 决策：
"先全流程跑一遍作为基准 然后准备对 ASW 这一个部分进行 C++ 和 RUST 化"。

### 算法 + 加速思路

三个 silhouette 共用同一个核心：**chunked pairwise euclidean distance**。
关键 trick：`‖xi − xj‖² = ‖xi‖² + ‖xj‖² − 2·xi·xj`。`X·Xᵀ` 用 BLAS
gemm 算（峰值 ~50 GFLOPS / 16 核），把 O(N²·D) 的 dist 计算从纯标量
循环（~minutes）压到 BLAS gemm（**~20 ms** for 222k × 20 dims）。

分块策略（M = 2048 chunk）：
- 内存上界：M² floats = 16 MB / chunk pair（极小，可 scale 到 N=1M）
- 对每个 (chunk_i, chunk_j)：BLAS gemm + 加 ‖·‖² 行/列向量 → 距离块
- 按 label 结构累加 per-cell (a, b) silhouette terms（rayon parallel）

### 实现选型

| 选项 | 优 | 劣 |
|------|----|----|
| **Rust + ndarray + ndarray-linalg (openblas-static)** | 与现有 `rust/crates/kernels` 一致；安全 | Windows openblas-static 编译有坑（要 vcpkg / 自带 dll）|
| Rust + matrixmultiply (pure Rust SIMD) | 0 build deps | 比 BLAS 慢 2-5×（pure Rust matmul 不达 cuBLAS / MKL） |
| C++ + pybind11 + Eigen MKL | Eigen 成熟，性能上限高 | 引入 C++ build system；与项目 Rust 主线不一致 |

**推荐方案**：**Rust + ndarray-linalg (openblas-static)**。理由：
1. 与项目 Rust 主线一致（GS-3 silhouette_precomputed 已在 `kernels` crate）
2. openblas-static 在 Windows 上一次配通就稳（用 vcpkg），CI 可固化
3. 性能上限与 Eigen+MKL 相当（OpenBLAS 与 MKL 在 sgemm 上差 < 20%）
4. 对照 `feedback_rust_speedup_assumption`：sgemm 是 BLAS 的强项，纯 Rust
   matmul 会输给 BLAS——所以**必走 BLAS 后端**，不要尝试 pure Rust SIMD

### 实现拆解（估 1-2 周）

| 任务 | 估时 | 备注 |
|------|-----:|------|
| Windows openblas-static / vcpkg 接通 | 1 天 | Linux/macOS 简单，Windows 是难点 |
| `rust/crates/kernels/src/asw.rs` 三函数实现 | 2 天 | label / batch / isolated 共享 chunked-dist primitive |
| PyO3 wrapper in `py_bindings` | 0.5 天 | pattern 跟 silhouette_precomputed 一致 |
| Cross-validation vs scib-metrics（pancreas + 222k） | 1 天 | 阈值 |Δ| < 1e-4 |
| 接到 `fast_auto_scrna/scib_metrics/scib.py` + try/fallback | 0.5 天 | scib-metrics 作为 fallback |
| 测试 + 文档 | 1 天 | 微基准 + ASW 数值一致性测试 |
| **总计** | **~6 天 + 缓冲** | |

### 预期收益

| 指标 | scib-metrics (current) | Rust + BLAS (target) | 加速 |
|------|---------:|---------:|----:|
| 222k label_asw | 130-140 s | **< 1 s** | ~150× |
| 222k batch_asw | 55 s | **< 5 s** | ~10× |
| 222k iso_label_asw | 140-160 s | **< 5 s** | ~30× |
| **Total per route** | **5.5 min** | **< 15 s** | **~22×** |
| **all-mode 4 routes** | **22 min** | **~1 min** | **~22×** |

**全管线收益**：222k all-mode wall 55 min → **~35 min**（−36%）。

### 风险 / Open questions

1. **Windows 端 openblas-static**：vcpkg 配置 + `link.exe` 找 `.lib` 路径；
   是否能避免动态依赖（`libopenblas.dll`）让 wheel self-contained？
2. **ASW 数值一致性**：scib-metrics 用 jax.numpy.float32，我们用 f32 BLAS；
   `silhouette_samples`-style per-cell 累加顺序不同→浮点漂移。可接受范围
   定 |Δ| < 1e-4（GS-3 给 silhouette_precomputed 是 3e-8 的级别，但那是
   precomputed dist 不需要重算 distance）。
3. **cosine 选项**：scib-metrics 默认 euclidean，但 atlas 上 PCA 后做
   cosine 可能更合理；先实现 euclidean parity，cosine 留 Phase 2。
4. **何时启动**：基线（v2-P10 222k all-mode）跑完确认 22 min ASW 真的
   是瓶颈再动手。如果 baseline 显示别的瓶颈（e.g. Phase 2b 远超估计），
   先解决那个。

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
5. ~~**graph-silhouette 度量审计**~~ **（2026-04-24 完成；2026-04-25 再次
   迭代到 knee picker，见 6）** —— 1000-cell 子采样在 222k kNN 图上只剩
   0.012% 连边，distance = 1-connectivity 99.89% 是 1，silhouette 纯噪声。
   换成 conductance 作为第一版 v2-P7 默认，picker 选 r=0.05 ARI 0.69，在
   atlas 数据上 OK。但 pancreas 轨迹型数据上 conductance argmin 仍会
   edge-pick k=3，不行。详见下一条 v2-P8/P9 演化。
6. ~~**knee picker + pipeline split**~~ **（2026-04-25 完成，v2-P8 + v2-P9，
   commit 199d1c6）**
   - pancreas (1000 cells, 8 SubCellType 真值) 暴露 conductance argmin
     的本质缺陷：picker 选 k=3 ARI 0.44，真值 r=0.30 k=7 ARI 0.65，
     clip `(3,10)` 默认正好套死最小边界。`leiden_target_n` 在 atlas
     上碰巧对是运气，不是度量本身的能力。
   - 用户（2026-04-25）关键反馈：首次聚类应 over-cluster 给 marker 注释
     后 merge，under-clustering 不可逆 → picker 应偏向多簇。
   - 先试 PCA-scree 式 perpendicular-line Kneedle（mirror 自 rust/
     kernels/src/pca.rs perpendicular_elbow），在 pancreas 多阶梯
     曲线上全局最大偏离打在中段 r=0.38 k=9，不合"视觉第一拐点"语义。
   - 换 `first_plateau_after_rise` 检测器：从 y[0] 出发向右扫，返回
     第一个满足 (a) 累计上升 ≥ 10% 总幅度 AND (b) 局部斜率 < 25%
     所见最大 的点。pancreas 上 r=0.25 k=7 ARI 0.53（匹配真值 k=7）。
   - 两段扫一度尝试过（coarse 50 点 + fine 14 点）：wall 38 min 但
     在 atlas 上和 single-stage 150 点 picks 不一致（harmony 差 2×），
     根因是滚动斜率 `window=5` 按索引计，不同网格密度下覆盖的
     resolution 宽度不同，算法对网格敏感。
   - v2-P9 最终方案：**拆 Phase 2**。2a 跑全路由 scIB（no Leiden，~1-2
     min），auto-select winner（scIB mean 最高），2b 只为 winner 跑
     single-stage 150-point Leiden + ROGUE + SCCAF。222k wall **34.5
     min**（vs all-routes single-stage 79 min，**2.3×** 加速，同等精度）。
   - 默认 `cfg.resolution_optimizer="knee"`，`knee_detector="first_plateau"`,
     `knee_two_stage=False`, `leiden_resolutions=arange(0.01,1.51,0.01)`,
     `cluster_method=None`（自动选）。
   - 16/16 pytest 过（含 4 个新 knee 测试）。12 张 plots 含
     `scib_heatmap_pre_cluster.png`（Leiden 前就出跨路由热图）。
7. ~~**rda → AnnData loader**~~ **（2026-04-25 完成）** —— `benchmarks/
   rda_to_h5ad.R` 用 R Seurat 包 dump 稀疏 mtx + obs csv + var txt；
   `benchmarks/assemble_pancreas_h5ad.py` 在 Python 端组装 AnnData 写
   `data/pancreas_sub.h5ad`。`benchmarks/smoke_pancreas.py` 跑完整
   pipeline on 1000 cells（~40s wall），picker 端到端验证。
8. ~~**v2-P9.1 CPU 礼让**~~ **（2026-04-25 完成）** —— 长 Leiden sweep 时
   前台（视频/浏览器）卡顿。改动：(a) `_leiden_sweep` 默认
   `max_workers = cpu_count - 4`（给 OS/前台留 4 核）；(b) worker
   initializer 在 Windows 走 ctypes `SetPriorityClass(BELOW_NORMAL)`、
   Unix 走 `os.nice(10)`，零新依赖。config 暴露 `max_leiden_workers`
   + `leiden_worker_priority` 两个 knob。
9. **OOM-1**：接入 `anndataoom` 的分块 preprocess（normalize / log1p /
   HVG / scale）—— 当前 scanpy 全内存，上不去更大的图谱。
10. **Harmony theta 调参**：222k 上 theta=4 收敛失败（iLISI=0.114），
    扫 theta∈{1,2,6,8} 看能不能救，救不回来就文档里定论 BBKNN 在这类
    atlas 的默认地位。
11. **graph-silhouette 度量审计（持续）**：conductance 和 first_plateau
    都依赖"曲线形状"。研究 stability-based（multi-seed ARI，前述实验
    222k bbknn 有清晰内部峰 r=0.10）能否做成 picker 的 hybrid 补充。

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

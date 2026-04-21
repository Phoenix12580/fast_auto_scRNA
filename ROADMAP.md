# ROADMAP — fast_auto_scRNA 整合对比平台演进

**当前版本基线**:v0.2.0(2026-04-22) · **上个基线**:v0.1.0(2026-04-21)

核心目标:**把整合 (batch integration) 的 Rust 加速 + 方法对比做到可发表的严谨程度**。

---

## ✅ v0.2 已完成(2026-04-22 release)

### 0.2.1 Pipeline 架构重写

- `integration: "none" | "bbknn" | "harmony" | "all"` 路线选择器替代老的 `run_bbknn/run_harmony` bool 对(老架构是 bug:`run_harmony=True` 下 Harmony 写出 `X_pca_harmony` 但 UMAP 根本没消费,直接吃 BBKNN 图)
- 两阶段主循环:Phase 1 (integration + UMAP) → 对比图**即刻落盘** → Phase 2 (scIB + Leiden + ROGUE + SCCAF + recall)
- 所有 per-route 产物按 method 后缀命名:`obsm[X_umap_<m>]`, `obsp[<m>_connectivities]`, `obs[leiden_<m>]`, `uns[scib_<m>]`, `uns[rogue_per_cluster_<m>]`
- HVG (seurat_v3 @ 2000) + scale (max_value=10) 补回流程(老架构主流程跳过,是 bug)
- 单 batch 输入自动降级到 `integration=none`(BBKNN/Harmony 无意义)

### 0.2.2 张泽民框架 9 指标 scIB 对齐

参照 *Zhang lab Cross-tissue fibroblast atlas, Cancer Cell 2024* 的三维评价:

| 维度 | 指标 | 实现来源 | Rust 加速 |
|---|---|---|---|
| **批次消除** | iLISI, kBET | scatlas.metrics | ✅ Rust |
| | batch silhouette (ASW) | sklearn | ⚠️ 纯 Python |
| **生物保留** | cLISI, graph_connectivity | scatlas.metrics | ✅ Rust |
| | label silhouette (ASW) | sklearn | ⚠️ 纯 Python |
| | **isolated label silhouette** (新)| sklearn 等价 scib_metrics | ⚠️ 纯 Python |
| **簇同质性** | **ROGUE** (新,Zhang lab) | scvalidate.rogue_py | ✅ Rust(entropy_table) |
| | **SCCAF** (新) | sklearn LR CV 等价 | ⚠️ 纯 Python |

**summary**:Batch = mean(3), Bio = mean(4), Homo = mean(2), Overall = 0.35·B + 0.45·Bio + 0.20·H

热图 `compare_scib_heatmap()` 自动在 `integration="all"` 出图时同步落盘。per-cluster ROGUE 单独存 `uns['rogue_per_cluster_<m>']`(用户 feedback 要求)。

### 0.2.3 UMAP 坍塌 bug 修复

发现原 `_pick_init` 在 PCA init 上用全局 `max|x|=10` 归一化,当 PC1 方差 >> PC2 时导致 **维度坍塌**(条带化)。修复路径:

1. 默认 `init='spectral'`(保险),委托 umap-learn 的 `spectral_layout`(bit-level 对齐)
2. 新增 `single_thread` 参数:关闭 Rust Hogwild SGD 的 rayon,parity 测试用
3. `parity_pancreas_umap.py` 脚本确认:同一 spectral init + 单线程 Rust vs umap-learn numba,Procrustes disparity **0.039**,trustworthiness 差 **0.004**

实测后发现:**新 kNN 默认 (cosine + k=30) 下 PCA init 也不再坍塌**,且与 Seurat/SCOP R 参考图拓扑一致。v0.2 最终默认 `umap_init='pca'`,spectral 保留作 fallback。

### 0.2.4 Harmony 2 迭代参数修正

原默认 `max_iter=10, theta=2` 在 2-batch atlas 上 **iter 2 就"收敛"**(实际是弱整合局部最优)。`ablate_harmony.py` 扫 4 档:

| 配置 | iLISI | kBET | converged iter |
|---|---|---|---|
| theta=2, max_iter=10 (old) | 0.086 | 0.068 | iter 2 |
| theta=2, max_iter=20 | 0.085 | 0.068 | iter 2(max_iter 没用)|
| **theta=4, max_iter=30** | **0.174** | **0.146** | iter 9 |
| dynamic-lambda | 0.048 | 0.037 | iter 4(更差)|

**结论:theta 是真正的杠杆,不是 max_iter**。v0.2 锁定默认 `harmony_theta=4.0`(atlas-scale 推荐),`harmony_max_iter=20`。

### 0.2.5 scIB silhouette 的 O(N²) 优化

`compute_silhouette: bool = True`(默认);关闭时 157k × 3 路线总耗时 **22 min → 8.5 min**(去掉三路的 O(N²) silhouette 计算)。

### 0.2.6 kNN + UMAP 默认对齐 SCOP

- `knn_n_neighbors: 30`(Seurat 默认)
- `knn_metric: "cosine"`(scRNA 标准;L2 归一化后走 bbknn Rust 欧氏核,数学等价)
- `umap_init: "pca"`(对齐 R `Seurat::RunUMAP`)

### 0.2.7 157k atlas 实测(v0.2 新基线)

| method | Batch | Bio | Homo | **Overall** |
|---|---|---|---|---|
| none | 0.00 | 0.98 | 0.78 | 0.60 |
| **bbknn** | **1.00** | 0.85 | 0.79 | **0.89** |
| harmony | 0.16 | 0.90 | 0.77 | 0.62 |

→ **BBKNN 夺冠**,与 Zhang lab Cancer Cell 2024 结论一致。总耗时 **13.5 min**(三路完整 9 指标 + Leiden 扫描 + ROGUE + SCCAF)。

---

## 🎯 v0.3 计划(下一版,Rust 加速优先级 P0)

### 0.3.1 scIB 补齐剩余 scib-metrics

v0.2 里张泽民九指标已完成,但 Luecken 2022 benchmark 标准集还差:

| 指标 | 类别 | 状态 | 工作量 |
|---|---|---|---|
| **NMI** (Normalized Mutual Info) | bio | ❌ | sklearn wrap,0.5 天 |
| **ARI** (Adjusted Rand Index) | bio | ❌ | sklearn wrap,0.5 天 |
| **Isolated label F1** | bio | ❌ | **Rust kernel 新增**,1 天 |
| **PCR** (Principal Component Regression) | batch | ❌ | **Rust kernel 新增**,1 天 |
| Cell cycle conservation | bio | ❌ | 需 marker 集,optional |

完成后 scIB 热图从 9 列扩到 13 列,Overall 权重按 Luecken 标准重算。

### 0.3.2 接入 scVI / Scanorama / fastMNN

覆盖 scib-benchmark 前 6 名:

| 方法 | 范式 | 实现 | 胶水难度 |
|---|---|---|---|
| **scVI** | 深度 VAE (ZINB) | scvi-tools (PyTorch GPU) | ~100 LOC |
| **Scanorama** | pairwise MNN + SVD | scanorama (纯 Py) | ~30 LOC |
| **fastMNN** | batchelor | rpy2 调 R,或 mnnpy | ~50 LOC |

统一成 `integration` 选项:`"scvi" / "scanorama" / "fastmnn"`。`"all"` 扩展到 6-way。

### 0.3.3 Rust HVG + scale

目前走 scanpy,seurat_v3 HVG 在 157k 上 ~20s。Rust 化预期 5-10×:
- `pp.highly_variable_genes` — seurat_v3 VST fit per-gene(rayon 并行)
- `pp.scale` — per-gene z-score + max_value clip(column-parallel,零拷贝)

### 0.3.4 Rust silhouette

sklearn silhouette 是 O(N²),157k 爆表。Rust 实现 + rayon 分块 + 可选 subsample,目标 atlas 级可开。

### 0.3.5 Rust Harmony vs R harmony parity 报告

- 装 R 4.x + harmony 2.0 在 WSL
- 同 seed 跑 panc8 / pancreas_sub 两个数据集
- 对比每细胞 embedding cosine similarity + 下游 Leiden ARI
- 目标:cos sim 中位数 > 0.95,下游 ARI > 0.90

---

## 🔭 v0.4+ 远期愿景

### 方法多样性

- **scANVI**(label-aware scVI,scib 并列 #1)
- **LIGER iNMF**(唯一非 deep 非线性范式)
- **Symphony**(reference mapping,基于 Harmony)
- **GLUE / Multigrate**(多模态 RNA+ATAC)

### 工程

- **Docker `fast_auto_scRNA:full`**:预打包 R + 所有 upstream 依赖,避免用户折腾 rpy2
- **GitHub Actions CI**:每次 push 跑 Rust/Python 测试 + pancreas smoke,每周自动 benchmark
- **GPU 路径**:scVI / scANVI 原生 PyTorch,CPU fallback 不动

### 科学严谨性

- **参数敏感度扫描**(Harmony theta × sigma、BBKNN k 网格),输出热图
- **Bio vs Batch Pareto 曲线**(审稿必看的"是否过度校正"判定)
- **论文级 benchmark 协议**(方法 × 指标 × 数据集 完整 CSV)

---

## 📊 与 SCOP 的战略定位

| | SCOP (R) | fast_auto_scRNA |
|---|---|---|
| 整合方法数量 | 15 | 3(v0.2),目标 6+(v0.3) |
| Rust 原生 | 0 | **2**(Harmony, BBKNN)|
| scIB 指标 | 部分 | **9**(Zhang 框架 @ v0.2),目标 13(Luecken @ v0.3)|
| 参数调优默认 | 全 upstream 默认 | **基于 ablation 锁定**(Harmony theta=4 etc.)|
| 速度(157k × 3 routes) | N/A | **13.5 min** |
| 可再现 benchmark | 手动 | **auto-heatmap + auto-plot** |
| 许可 | MIT | MIT |

**定位**:不是 SCOP 的替代(不追 15 方法全覆盖),而是 **2-3 个核心方法做到 Rust 加速 + Zhang 框架级对比严谨**。

---

## 📂 文档输出计划

v0.3 完成后产出:

- `docs/methods.md` — 可直接粘贴进论文 Methods
- `docs/benchmark_v0.3.md` — 完整方法 × 指标 × 数据集报告
- `docs/harmony_parity.md` — Rust Harmony ↔ R harmony 一致性证明
- `docs/scib_metrics_derivation.md` — 我们 Rust scib-metrics vs Python `scib-metrics` 包 parity

**投稿目标**:Bioinformatics Applications Note 或 Nature Methods Brief。

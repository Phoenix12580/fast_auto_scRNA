# ROADMAP — fast_auto_scRNA 后续优化与科研严谨性计划

**状态日期**:2026-04-21 · 版本基线:v0.1.0

本文档列出 v0.1 之后的优化路线,**核心目标:把整合(batch integration)环节做到可发表的严谨程度**。

起因:对比 SCOP 时发现该包的整合支持面其实很全(15 种方法),但每个都是 upstream R/Python 包的薄壳 wrapper,**默认参数都没动**,且**缺乏统一的 scib-metrics 对比输出**。我们要做的不是重写这些算法,而是:
- **在 fast_auto_scRNA 里做一个一键的多方法对比平台**
- **把 Harmony + BBKNN 的 Rust 实现做到比原版快但结果一致**
- **提供符合 Luecken 2022 scib benchmark 标准的严谨评估**

---

## 1. 现状诊断

### 1.1 整合方法当前覆盖

| 来源 | 数量 | 已在 fast_auto_scRNA |
|---|---:|---|
| SCOP 支持 | 15 | — |
| scatlas Rust native | **2** | Harmony 2.0, BBKNN |
| 通过 Python 可接 | — | 需加胶水 |

### 1.2 SCOP 整合方法全景(来自 `integration.R` + `integration5.R` 扫描)

| 方法 | 范式 | 输出 | CPU/GPU | Luecken 2022 scib 排名 | Rust 化? |
|---|---|---|---|---:|---|
| **scVI** | 深度 VAE(ZINB)| embedding | GPU 优先,CPU ≤ 50k | **#1** 综合 | ❌ 用 scvi-tools |
| scANVI | scVI + 标签监督 | embedding | GPU 优先 | **#1** 综合 | ❌ 同上 |
| **Harmony** | 线性(soft k-means + MoE ridge)| embedding | CPU | **#2-3** 综合 | ✅ **scatlas (6× R)** |
| Scanorama | 线性(全对 MNN + SVD)| embedding | CPU | #4 | ❌ Python 调用 |
| **BBKNN** | 图论(batch-balanced kNN)| 只给图 | CPU | #5(批次去除强)| ✅ **scatlas (15× + HNSW)** |
| fastMNN | 线性(batchelor)| embedding | CPU | #6 | ❌ R 调用 |
| LIGER iNMF | 线性(整合 NMF)| embedding | CPU | #7 | ❌ R 调用 |
| CSS | 线性(cluster-similarity spectrum)| embedding | CPU | 未在 scib | ❌ |
| Coralysis | 非线性(迭代聚类投影)| embedding | CPU | 未在 scib | ❌ |
| Seurat v5 CCA/RPCA | 线性(anchors)| embedding | CPU,大 >100k 吃不动 | Seurat v5 标准 | ❌ |
| MNN 经典 | 计数校正 | counts | O(n²),>50k 不现实 | 被 fastMNN 替代 | 跳过 |
| Conos | 图论(跨 panel kNN)| graph | CPU,>100k 吃力 | scib 差 | 跳过 |
| ComBat | 经验贝叶斯 | counts/PCs | CPU | 过度校正丢 bio | 跳过 |
| Uncorrected | 基线 | PCA | — | baseline | ✅ 已有 |
| Seurat v4 | 旧 anchor 路径 | counts | deprecated | 被 v5 替代 | 跳过 |

### 1.3 SCOP 里没有但 2024-2026 SOTA 应该加的

| 方法 | 作者/年 | 为啥要加 | 来源 |
|---|---|---|---|
| **scANVI**(label-aware scVI)| scverse | **scib #1**,SCOP 只有 scVI | scvi-tools |
| Symphony | Korsunsky lab | reference mapping,CPU,速度 > scArches | R 包 |
| scArches / scPoli | scverse | atlas query 标准 | scvi-tools |
| Harmony2 / Harmony 2.0 | Korsunsky 2023 | **我们 scatlas 就是移植这个** — 明确声明 | pati-ni/harmony |

### 1.4 严谨性的具体缺口

| 评审问题 | 我们当前回答 | 缺口 |
|---|---|---|
| 你比了几种整合方法? | 1 种(Harmony)| **至少 5-6 种才能写 benchmark 论文** |
| 你用了 Luecken 2022 scib 全部指标吗? | 部分(iLISI/cLISI/graph_conn/kBET/silhouette)| 缺 **NMI / ARI / isolated label F1 / PCR(principal component regression)** |
| 你评估了 over-correction 吗? | 没 | 缺 **biology conservation vs batch removal 权衡曲线** |
| scib 分数怎么聚合? | 单个 mean | 缺 **weighted sum**(Luecken 格式)|
| 参数是否调优过? | 全默认 | 每个方法应该做 **小范围参数扫描** |
| 你的 Rust Harmony 跟 R 原版结果一致吗? | 已验(Rust 自测 2-cluster 分离)| 缺 **vs R harmony 逐细胞 ARI 对比** |
| BBKNN 在图论方法中为啥选它? | 没解释 | 缺 **vs Conos / umap-learn kNN 对比** |

---

## 2. 优先级排序的优化路线

按 ROI(严谨性提升 ÷ 工作量)排序。

### P0 — 必做,v0.2 (严谨性底线)

#### P0.1 多整合方法一键对比(5-6 种)

把 SCOP 的 `integration_scop` 思路搬进 fast_auto_scRNA,封装成:

```python
scatlas_pipeline.compare_integrations(
    adata,
    methods=["harmony", "bbknn", "scvi", "scanorama", "fastmnn", "uncorrected"],
    batch_key="batch",
    label_key="celltype",  # 用于 bio conservation 评估
)
# 返回 DataFrame:method × scib_metric
```

优先级:
1. **Harmony**(已有,Rust)
2. **BBKNN**(已有,Rust)
3. **Uncorrected**(baseline,已有)
4. **scVI**(要装 scvi-tools,GPU optional,~100-150 LOC 胶水代码)
5. **Scanorama**(要装 scanorama,CPU,~30 LOC 胶水)
6. **fastMNN**(要 R + batchelor 或者用 Python port `mnnpy`,先用 R)

**工作量**:3 天(胶水 + 统一 API)
**科学价值**:审稿问"你和其他方法比了吗" → "比了 6 种,包括 scib 公认第一名 scVI"

#### P0.2 完整 scib-metrics 指标

目前 scatlas.metrics 有 **iLISI / cLISI / graph_connectivity / kBET / silhouette**。
补齐到 Luecken 2022 完整集合:

| 指标 | 类别 | 现状 | 需要 |
|---|---|---|---|
| iLISI | batch | ✅ Rust | — |
| cLISI | bio | ✅ Rust | — |
| graph_connectivity | batch | ✅ Rust | — |
| kBET | batch | ✅ Rust | — |
| silhouette(batch + label)| both | ✅ sklearn wrap | — |
| **NMI** | bio | ❌ | sklearn 直接有,wrap |
| **ARI** | bio | ❌ | sklearn 直接有,wrap |
| **Isolated label F1** | bio | ❌ | **新 Rust kernel**,~100 LOC |
| **PCR** (principal component regression) | batch | ❌ | **新 Rust kernel**,闭式解简单 |
| **Cell cycle conservation** | bio | ❌ | 需要 marker 基因集 + 可选 |

**工作量**:1.5 天
**科学价值**:可以直接声称 "我们用了 Luecken 2022 同套指标",严谨性即被认可

#### P0.3 方法 × 指标 × 数据集 完整 benchmark 表

输出 `benchmarks/integration_benchmark_v0.2.csv`,格式:

```
method,dataset,n_cells,iLISI,cLISI,graph_conn,kBET,NMI,ARI,isoF1,PCR,silh_batch,silh_label,mean,time
Harmony,panc8,1600,1.00,0.82,0.90,1.00,0.71,0.62,0.85,0.12,0.38,0.45,0.74,14s
BBKNN,panc8,1600,...
scVI,panc8,1600,...
```

数据集:**panc8_sub** (小,5 tech) + **pancreas_sub** (单样本基线) + **epithelia_157k** (大规模)

**工作量**:跑完 P0.1 + P0.2 后自然产出,封装 2 天
**科学价值**:Methods 部分的核心 table

#### P0.4 Rust Harmony vs R harmony parity 报告

当前我们 Rust Harmony 只做了 synthetic 验证。生产级论文需要:
- 固定 seed
- 同一数据(panc8 + pancreas_sub)
- R harmony 2.0 和 scatlas Harmony 各跑一遍
- 对比:每细胞 embedding 的 **cosine similarity** + 基于 embedding 的下游聚类 **ARI**
- 预期:cosine > 0.95,聚类 ARI > 0.90

**工作量**:需要先在 WSL 装 R + harmony。1-2 天。
**科学价值**:直接答"你的 Rust port 跟原版结果一致吗"

### P1 — 推荐做,v0.3 (方法多样性)

#### P1.1 加 scANVI、Symphony、LIGER

- **scANVI**:scib #1,SCOP 没有,加上能进 scverse 生态
- **Symphony**:reference mapping,CPU,速度 > scArches
- **LIGER iNMF**:iNMF 范式(唯一的非 deep 非线性),代表性

**工作量**:3 天(每个 ~1 天胶水)

#### P1.2 参数敏感度扫描

对 Harmony 扫 theta ∈ {1, 2, 4, 8} × sigma ∈ {0.05, 0.1, 0.2}
对 BBKNN 扫 k ∈ {3, 5, 10}
输出 heatmap:参数 → scib mean

**工作量**:1 天(用 P0.3 框架跑多轮)

#### P1.3 Biology / Batch 权衡曲线

- X 轴 = batch removal(iLISI)
- Y 轴 = bio conservation(NMI)
- 散点 = 各方法 + 各参数
- Pareto 前沿显示最优权衡

**工作量**:0.5 天(画图)
**科学价值**:审稿必看的"是否过度校正"判定图

### P2 — 长线,v0.4+

#### P2.1 多模态(RNA + ATAC)

- **GLUE**(gao-lab)或 **Multigrate**(scverse)
- 大多数 benchmark 用户只有 RNA,优先级低

#### P2.2 Reference mapping(atlas query)

- **Symphony**(R,CPU)或 **scArches**(Python,GPU)
- 场景:用户有已建好的 atlas,要把新数据映射进去
- **工作量**:1 周

#### P2.3 GPU 路线

- scVI + scANVI + PyTorch
- 仅在有 GPU 时开,CPU fallback 已够用
- 记录 GPU 加速比(预期 5-20×)

---

## 3. 科研级比较 Benchmark 协议

下一个交付是 `scatlas_pipeline/benchmarks/integration_rigorous.py`,格式:

### 3.1 主表:方法 × 指标

按 Luecken 2022 Table S3 风格:

| 方法 | iLISI ↑ | cLISI ↑ | graph_conn ↑ | kBET ↑ | NMI ↑ | ARI ↑ | isoF1 ↑ | PCR ↓ | silh_batch ↑ | silh_label ↑ | **Batch**(mean 左)| **Bio**(mean 右)| **Total** | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Uncorrected | 0.12 | 0.98 | ... | ... | ... | ... | ... | ... | ... | ... | 0.15 | 0.85 | 0.43 | 3s |
| Harmony | 1.00 | 0.82 | 0.90 | 1.00 | ... | ... | ... | ... | ... | ... | 0.92 | 0.76 | 0.82 | 14s |
| BBKNN | ... | | | | | | | | | | | | | |
| scVI | | | | | | | | | | | | | | |
| Scanorama | | | | | | | | | | | | | | |
| fastMNN | | | | | | | | | | | | | | |

每个数据集(panc8 / pancreas / 157k)一张表。

### 3.2 权衡图

每个数据集的 bio vs batch 散点,Pareto 前沿标出。

### 3.3 参数扫描热图

Harmony theta × sigma 的 scib mean 热图(至少 panc8 上做)。

### 3.4 Rust Harmony ↔ R harmony parity 表

| 数据集 | cos sim 中位数 | 聚类 ARI | 时间 Rust | 时间 R | 加速比 |
|---|---|---|---|---|---|

---

## 4. 工程侧配套

### 4.1 新增 Rust kernels

| Kernel | 用途 | 优先级 |
|---|---|---|
| `scatlas.metrics.nmi_ari` | NMI / ARI 聚类对比 | P0.2 |
| `scatlas.metrics.isolated_label_f1` | 每个小 cluster 的 isolation F1 | P0.2 |
| `scatlas.metrics.pcr` | principal component regression | P0.2 |
| `scatlas.metrics.cell_cycle_conservation` | 细胞周期保真度(可选)| P0.2+ |

### 4.2 新增 Python 胶水(非 Rust,调 upstream)

| 文件 | 内容 | 优先级 |
|---|---|---|
| `scatlas_pipeline/integrations/scvi_wrapper.py` | scvi-tools 调用 | P0.1 |
| `scatlas_pipeline/integrations/scanorama_wrapper.py` | scanorama 调用 | P0.1 |
| `scatlas_pipeline/integrations/fastmnn_wrapper.py` | rpy2 调 batchelor | P0.1 |
| `scatlas_pipeline/integrations/liger_wrapper.py` | rpy2 调 rliger | P1.1 |
| `scatlas_pipeline/integrations/symphony_wrapper.py` | rpy2 调 symphony | P1.1 |

统一接口:
```python
class IntegrationMethod(Protocol):
    name: str
    requires_gpu: bool
    def run(self, adata, batch_key, **kwargs) -> np.ndarray:
        """Return (n_cells, n_dims) embedding."""
```

### 4.3 新增依赖(可选,按方法激活)

- `scvi-tools`(PyTorch)— P0.1
- `scanorama`(纯 Python)— P0.1
- `rpy2` + R + `batchelor` + `rliger` + `symphony` — P0.1/P1.1

**Docker 镜像**:v0.3 里打一个 `fast_auto_scRNA:full` 镜像,把所有依赖打包,避免每个用户自己折腾 R 装。

### 4.4 CI

v0.2 里加 GitHub Actions:
- Rust: `cargo test --release` on ubuntu-latest
- Python: `pytest python/tests/`
- smoke test:`./setup.sh` 完整跑通
- 可选:每周 benchmark 自动跑,结果提交 `benchmarks/history.csv`

---

## 5. 推荐立即做的 3 个小动作(本周内)

1. **P0.2 NMI + ARI + isolated label F1 指标**(1 天)— 直接扩充 `scatlas.metrics`,纯 Rust kernel,为 benchmark 打底
2. **P0.1 加 scVI wrapper**(1 天)— 最简单的非 Rust 整合方法胶水,立即能跑对比
3. **P0.3 一键 benchmark 脚本**(1 天)— 跑完出表,直接可贴论文

**这三个做完,就足以写论文 Methods 部分回应 "为什么选你们的 Harmony + BBKNN 组合"**:
- 我们在 panc8 / pancreas / 157k 三个数据集上跑了 Harmony + BBKNN + scVI + Scanorama + fastMNN + Uncorrected
- 用了 Luecken 2022 完整 scib-metrics 指标
- scatlas Rust Harmony 跟 R harmony cosine > 0.95,速度 6×
- 全流程比 scanpy + upstream 快 20-30×

---

## 6. 与 SCOP 的战略定位

| | SCOP | fast_auto_scRNA |
|---|---|---|
| 整合方法数量 | 15 | 6 主要(+ 扩展)|
| Rust 原生 | 0 | 2(Harmony, BBKNN)|
| scib-metrics 完整 | 部分(仅画图)| **目标 v0.2 完整对齐 Luecken 2022** |
| 参数敏感度扫描 | 无 | **目标 v0.3** |
| GPU 支持 | scVI 可 | scVI 可 |
| **速度(157k)** | N/A(R 生态)| **3-4 min** |
| 可再现 benchmark 协议 | 无 | **目标 v0.2 发布** |

**定位口径**:
> "fast_auto_scRNA 不是要替换 SCOP,而是在 2-3 个核心方法(Harmony、BBKNN)上做 Rust 重写达到 6-15× 加速,同时提供 Luecken 2022 scib-metrics 完整的、可再现的方法对比平台 — 即 SCOP 的'全面但每个都浅'对比我们'精但深 + 严谨评估'。"

---

## 7. 文档输出计划

v0.2 + v0.3 完成后产出:
- `scatlas_pipeline/docs/methods.md` — 可直接粘贴进论文 Methods
- `scatlas_pipeline/docs/benchmark_v0.3.md` — 整合 benchmark 完整报告(方法 × 指标 × 数据集)
- `scatlas_pipeline/docs/scib_metrics_derivation.md` — 我们 scib-metrics 的 Rust 实现 vs scib-metrics Python 包的 parity 证明
- 投稿目标:Bioinformatics Applications Note(2 页快讯)或 Nature Methods Brief(整合+速度)

## 8. 与 plans/ 历史计划的关系

此 ROADMAP 替代 `plans/happy-sparking-falcon.md` 的 M8+。历史 plan 的 M3 已完成(scib metrics),但当前版本只做到 4 个指标,v0.2 会扩展到 Luecken 完整集合。

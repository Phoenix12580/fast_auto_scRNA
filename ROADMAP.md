# ROADMAP — fast_auto_scRNA 后续优化与科研严谨性计划

**状态日期**:2026-04-21 · 版本基线:v0.1.0

本文档列出 v0.1 之后的优化路线,**重点是把聚类环节做得足够严谨、可发表**。起因:对比 SCOP 时发现该包的聚类本身只是 Seurat `FindClusters` 的轻薄壳(louvain/slm/leiden,默认 res=0.6,k=20),**没有任何稳定性、共识、子聚类、silhouette/DB/CH、scSHC、chooseR、clustree、SC3 等验证/选模块**。要做一个科学上严谨的流程,必须补上这些。

---

## 1. 现状诊断

### 1.1 已有的组件

| 层 | 组件 | 实现 | 科学严谨性 |
|---|---|---|---|
| 归一化 | log1p + library size | scipy | 标准 |
| HVG | scatlas 无,用户侧 seurat vst | — | 标准 |
| PCA | 随机 SVD + Gavish-Donoho 自动维度 | Rust | **✓ MSE-最优硬阈值**,有理论基础 |
| 整合 | Harmony 2.0 / BBKNN / (未加 scVI / fastMNN) | Rust port 1:1 | Harmony 已达 paper + C++ 实现 |
| 聚类 | Leiden(scanpy flavor=igraph)+ Louvain(SCOP-aligned)| igraph C++ | **⚠ 单分辨率、无稳定性** |
| 聚类验证 | scvalidate recall(knockoff-filtered wilcoxon)| Rust | **✓ 有 FDR 控制** |
| 指标 | scib(iLISI/cLISI/graph_conn/kBET)+ silhouette(sklearn)| Rust + sklearn | 整合评估 ≠ 聚类质量评估 |
| UMAP | layout Hogwild SGD | Rust | 与 umap-learn trustworthiness 等价 |

### 1.2 科学严谨性的具体缺口

把同行评审视角套进来,**聚类**的严谨性评价通常问这几个问题:

| 评审问题 | 我们当前回答 | 缺口 |
|---|---|---|
| 你怎么选的 resolution? | "target_n=(8, 30) 范围扫 6 个固定值取第一个落进去的" | **启发式,没数据驱动**,resolution 敏感度未评估 |
| 不同 resolution 的聚类结果怎么汇报? | 只给最终一组 | **缺 clustree** 多分辨率可视化 |
| cluster 是否稳定?bootstrap/扰动下一致性多少? | 没测 | **缺 bootstrap 共识** |
| 为什么是 Leiden 而不是 SLM / Louvain / SC3 / phenograph? | 没对比 | **缺多算法 panel** |
| 每个 cluster 的 "纯度" 如何量化? | 目前只有 scib 整合评分 | **缺 silhouette / DB / CH / ROGUE 每簇质量** |
| cluster 数量是否显著高于 null? | 没测 | **缺 SC3 / scSHC 的 permutation null 对比** |
| 亚群(SubCluster)是否合理? | 没做 | **缺递归子聚类 + FDR** |

### 1.3 与 SCOP 的定位关系

SCOP 聚类仅是 Seurat 的包装,**它本身不具科学聚类严谨性**。以 SCOP 为基线做对比只证明速度,不证明严谨性优势。要声称 **"科研级严谨"**,我们要超越 SCOP,引入更多论文级方法。

---

## 2. 优先级排序的优化路线

按 ROI(严谨性提升 ÷ 工作量)排序。

### P0 — 必做,v0.2 (严谨性底线)

#### P0.1 多分辨率 Leiden + clustree 可视化

- 自动跑一组 resolution(例如 0.1, 0.2, ..., 2.0)
- 输出 `clustree` 风格的多分辨率树,展示 cluster 分合关系
- **工作量**:1 天。`clustree` 有 R 原版和 Python port。可以直接调 Python `clustree` 或自写。
- **科学价值**:审稿人首先问 "你扫了 resolution 吗",这个图是标准答案。

#### P0.2 Bootstrap / Sub-sampling 稳定性分析

- 对同一 Leiden 配置,对细胞做 90% bootstrap 采样 20 次
- 每次跑 Leiden → 算 cluster-wise ARI 稳定性 + per-cell 稳定性分布
- 输出:per-cluster 稳定性得分 + 整体 ARI 分布
- **工作量**:2 天。kNN 图可复用(只变细胞子集),Leiden 调用便宜。
- **科学价值**:直接回答 "这个 cluster 会不会是采样噪声"。

#### P0.3 Silhouette / Davies-Bouldin / Calinski-Harabasz 每簇质量

- 在 PCA/Harmony 空间上算每个 cluster 的 silhouette / DB / CH
- 输出表格 + boxplot
- **工作量**:0.5 天(sklearn 直接有,rayon 可加速 silhouette 到 157k level)
- **科学价值**:最经典的 internal cluster validation 三件套,审稿必看

#### P0.4 ROGUE 每簇纯度

- `scvalidate.rogue_py.calculate_rogue` 已 Rust 实现(M1.3)
- 集成到 pipeline,输出 per-cluster ROGUE 得分
- **工作量**:0.5 天(接线)
- **科学价值**:单细胞领域专用的"簇是否单一态"度量,中国/华人 PI 圈子非常重视

### P1 — 推荐做,v0.3 (跨方法对比严谨)

#### P1.1 多聚类算法 panel

增加至少 3 种聚类算法的一键对比:

| 算法 | 来源 | 预期 ROI |
|---|---|---|
| Leiden (RB-modularity) | leidenalg(已有) | 基线 |
| Leiden (CPM) | leidenalg `partition_type="CPMVertexPartition"` | 不同 quality function 的差异 |
| Louvain | scanpy/igraph | SCOP 默认 |
| SLM (Smart Local Moving) | Seurat algorithm=3 via R 或重写 | SCOP 第二选项 |
| Phenograph | `phenograph` Python 包 | 流式流行 |
| **scSHC** | R 包(不 Rust 化,已在 memory 判了) | 严格 hierarchical + FDR |
| **chooseR** | R 包,bootstrap 稳定性选 resolution | 论文引用多 |

Panel 输出:**per-cell 聚类分配一致性矩阵**(Cell × Method, ARI 跨方法)。

- **工作量**:1 周
- **科学价值**:审稿人问 "你和其他方法比了吗" → 答 "比了 6 种" → 稳

#### P1.2 Cluster merging by DEG Jaccard(atlas_pipeline/05b 的 Rust 化)

- Leiden 后自动算每簇 top-50 DEG(`scatlas.stats.wilcoxon_ranksum_matrix`,已 Rust)
- 计算簇间 Jaccard overlap,超阈值自动合并
- 迭代直到稳定
- **工作量**:0.5 天(逻辑已在 atlas_pipeline/05b,搬进 scatlas.tl 即可)
- **科学价值**:明确 "避免过分拆簇" 的方法依据,可在方法部分写清楚

#### P1.3 递归子聚类(SubCluster)

- 对大 cluster(e.g. > 10% of cells)自动递归 Leiden
- 子 cluster 用 recall/knockoff FDR 验证
- **工作量**:2 天
- **科学价值**:发现亚型的标准做法

### P2 — 长线,v0.4+

#### P2.1 Permutation null / SC3 风格的 cluster 显著性

- 打乱基因标签,重跑聚类,算 null distribution 下 cluster 数量分布
- 显著性 p-value: 真实 cluster 数 > 99% of null
- **工作量**:3 天(需要并行 N 次独立 pipeline,大数据昂贵)

#### P2.2 GPU 路线(可选)

- Rapids-cuML UMAP / Louvain(NVIDIA)
- scVI(PyTorch GPU)
- 仅当用户有 GPU 才开启
- **工作量**:1 周
- **注意**:MEMORY 已记录 "生产服务器无 GPU,主加速走 CPU,GPU 只做 optional"。符合规划。

#### P2.3 scVI / fastMNN / LIGER 多整合对比

- 把 atlas_pipeline/03_integrate_bench 的思路接进 scatlas_pipeline
- 5 方法并跑,每个出 scib 分数
- **工作量**:1 周(主要是依赖安装 + 接口对齐)

---

## 3. 科研级比较 Benchmark 协议(核心交付)

下一个里程碑应该产出 **"benchmark_rigorous.py"**,跑出以下表格格式结果:

### 3.1 聚类方法 × 数据集矩阵

| 数据集 | 聚类方法 | N clusters | ARI vs celltype | ARI vs SubCellType | Silhouette | DB Index | CH Index | ROGUE mean | 稳定性 (20×bootstrap ARI) | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|
| panc8_sub | Leiden (RB-mod, r=0.5) | 9 | 0.62 | 0.48 | 0.43 | 1.2 | 450 | 0.82 | 0.91 ± 0.04 | 14s |
| panc8_sub | Leiden (CPM) | | | | | | | | | |
| panc8_sub | Louvain (r=0.6) | | | | | | | | | |
| panc8_sub | SLM (r=0.6) | | | | | | | | | |
| panc8_sub | Phenograph | | | | | | | | | |
| panc8_sub | scSHC | | | | | | | | | |
| panc8_sub | recall-validated | | | | | | | | | |
| pancreas_sub | ... | | | | | | | | | |
| epithelia_157k | ... | | | | | | | | | |

### 3.2 Resolution 敏感度扫描(每个数据集)

- clustree 图(从 r=0.1 到 r=2.0)
- 每分辨率的 ARI / silhouette / ROGUE 曲线

### 3.3 Bootstrap 稳定性(每个方法)

- N=20 次 90% 下采样
- 汇报 per-cluster 稳定性 + 整体 ARI 分布 箱线图

### 3.4 消融实验

- 无整合 vs Harmony vs BBKNN vs Harmony+BBKNN
- 每个 ablation 跑上述完整验证

---

## 4. 工程侧配套

### 4.1 新增 Rust kernels

| Kernel | 用途 | 优先级 |
|---|---|---|
| `scatlas.metrics.silhouette_rayon_f32` | per-cell silhouette 并行,157k 能跑 | P0.3 |
| `scatlas.metrics.davies_bouldin` | 每簇 DB index | P0.3 |
| `scatlas.metrics.calinski_harabasz` | CH index | P0.3 |
| `scatlas.tl.bootstrap_leiden` | 并发 N 次 Leiden 采样 | P0.2 |
| `scatlas.tl.clustree_matrix` | 跨 resolution cluster 分配矩阵 | P0.1 |
| `scatlas.tl.sub_cluster` | 递归 Leiden + 子层 recall 验证 | P1.3 |
| `scatlas.tl.merge_clusters_by_deg` | DEG Jaccard 合并 | P1.2 |

### 4.2 新依赖(可选,按方法需要)

- `phenograph`(Python)— P1.1
- 本地 R + scSHC / chooseR(Docker 镜像化,避免装 R 到 WSL)— P1.1
- `clustree` R 或 Python port — P0.1

### 4.3 CI

目前无 CI。v0.2 里加 GitHub Actions:
- Rust: `cargo test --release` on ubuntu-latest
- Python: `pytest python/tests/`
- smoke test:`./setup.sh` 完整跑通

---

## 5. 推荐立即做的 3 个小动作(本周内)

1. **P0.3 silhouette/DB/CH + P0.4 ROGUE 接线**(1 天)— 立即显著提升严谨性,工作量小
2. **P0.1 clustree 多分辨率图**(1 天)— benchmark 里就能出标准图
3. **P1.2 DEG Jaccard merge**(0.5 天)— 已有代码参考,纯迁移

**这三个做完,就足以回应 "你这个流程在聚类严谨性上跟 SCOP / Seurat 区别在哪"**:
- 你有 silhouette + DB + CH + ROGUE(SCOP 都没有)
- 你有 clustree(SCOP 没有)
- 你有自动 DEG merge(SCOP 没有)
- 你有 recall-validated clustering(SCOP 没有)
- 速度还快 20-30×(已证明)

---

## 6. 与 plans/ 历史计划的关系

此 ROADMAP 替代 `plans/happy-sparking-falcon.md` 的 M8+,后者仅列了 leiden wrapper。当前规划更聚焦 **科研严谨性** 而非单纯速度。

## 7. 文档输出计划

当 P0 + P1 完成后,产出:
- `scatlas_pipeline/docs/methods.md` — 可直接粘贴进论文 Methods 部分的英文描述
- `scatlas_pipeline/docs/benchmark_v0.3.md` — 含所有表格和图的 benchmark 报告
- 投稿目标:Bioinformatics Applications Note(单篇 2 页)+ 长版 BMC Bioinformatics

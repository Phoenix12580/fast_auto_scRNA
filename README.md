# fast_auto_scRNA

**一键自动化 scRNA-seq 分析流程 — Rust 加速,Python API**

端到端覆盖 `load → QC → 归一化 → PCA → BBKNN (可选) → Harmony (可选) → UMAP → Leiden → recall (可选) → scib metrics → per-cluster ROGUE`,所有热路径用 Rust 重写,同等质量下比纯 Python 流程快 **20-45×**。

---

## 特性对照

| 步骤 | Python 原生 | fast_auto_scRNA | 加速 |
|---|---|---|---|
| PCA(稀疏 SVD)| sklearn TruncatedSVD | Rust 随机 SVD + Gavish-Donoho 自动维度 | 2.5× |
| BBKNN 邻居 | `bbknn` Python | Rust HNSW + rayon | 15× |
| BBKNN 连接矩阵 | umap-learn `fuzzy_simplicial_set` | Rust 并行融合 | 3× |
| Harmony 2.0 | `harmonypy` | Rust 融合 update_R(6× R)| 6× |
| UMAP 布局 | `umap-learn`(numba 单线程)| Rust Hogwild SGD(rayon 16-core)| 16.75× |
| scib metrics(iLISI/cLISI/graph_conn/kBET)| `scib-metrics` | Rust 并行 | 5-10× |
| recall-validated 聚类 | pure Python wilcoxon | Rust 并行 wilcoxon + knockoff | 30-50× |
| per-cluster ROGUE 纯度 | pure Python entropy | Rust `entropy_table` + loess | 10-20× |

**端到端实测**(panc8 1600 cells × 12940 genes × 5 techs):
- 原始 scanpy + pure-Python recall:**425s**
- fast_auto_scRNA:**14s**(30.8× 加速)

**端到端实测**(epithelia 157k cells × 16337 genes × 2 batches):
- 不带 recall:**3.3 min**(206s,含 24s IO + 14s 归一化)
- scib mean 整合评分 **0.925**

---

## 架构

仓库由三个子包组成,协同工作:

```
fast_auto_scRNA/
├── scatlas/              Rust 核心:PCA / BBKNN / Harmony / UMAP / fuzzy / scib metrics
├── scvalidate_rewrite/   Rust 核心:wilcoxon / knockoff-filtered recall 聚类验证
└── scatlas_pipeline/     一键 Python API:串接上两者,暴露 `run_pipeline()`
```

依赖关系:`scatlas_pipeline` → `scatlas` + `scvalidate`。每个 Rust 子包发布 PyO3 wheel,Python 端 import 即用。

---

## 一键安装(Linux / WSL2)

```bash
git clone https://github.com/Phoenix12580/fast_auto_scRNA.git
cd fast_auto_scRNA
./setup.sh
```

`setup.sh` 会自动:
1. 检查 Rust 工具链(没装就提示 `rustup` 命令)
2. 检查 Python ≥ 3.10,创建 `.venv/`
3. 装 maturin + 科学栈依赖
4. 依次构建 `scatlas` + `scvalidate_rust` + 装 `scvalidate` Python 包 + 装 `scatlas_pipeline`
5. 跑 smoke test 验证安装成功

耗时约 **5-10 min**(首次编译 Rust)。

详细说明见 [INSTALL.md](INSTALL.md)。

---

## 快速使用

### Python API

```python
from scatlas_pipeline import run_pipeline

adata = run_pipeline(
    "data/my_sample.h5ad",
    batch_key="batch",         # 或 None(单样本)
    run_bbknn=True,            # batch-balanced kNN 图(多 batch 时建议开)
    run_harmony=True,          # Harmony 2.0 embedding 去批次
    run_recall=True,           # knockoff 验证聚类(≤ 10k cells 建议开)
    run_rogue=True,            # per-cluster ROGUE 纯度(默认开)
    use_anndataoom=False,      # 1M+ 细胞 densify OOM 守护(需先装包)
    out_h5ad="atlas.h5ad",
)
print(f"clusters: {adata.obs['leiden'].nunique()}")
print(f"scib mean: {adata.uns['scib_score']['mean']:.3f}")
print(f"ROGUE per-cluster: {adata.uns['rogue']['per_cluster_mean']}")
```

### YAML 配置文件

```bash
python -c "
from scatlas_pipeline import run_from_config, PipelineConfig
import yaml
cfg = PipelineConfig(**yaml.safe_load(open('my_config.yaml')))
run_from_config(cfg)
"
```

配置示例见 `scatlas_pipeline/configs/epithelia_157k.yaml`。

---

## 流程配置开关(PipelineConfig 字段对应)

| 字段 | 默认 | 场景 |
|---|---|---|
| `run_bbknn` | `True` | 多 batch + tech 严重失衡 → 开;单样本或 balanced batch → 关(走标准 scanpy cosine kNN) |
| `run_harmony` | `True` | 跨数据集 / 批次 → 开;single-sample → 关 |
| `run_umap` | `True` | 关掉节省 ~5s,但失去 `obsm['X_umap']` |
| `run_leiden` | `True` | 关掉就不聚类(但还会出 UMAP)|
| `run_recall` | `False` | **≤ 10k 细胞**建议开(scvalidate knockoff 验证),大数据 O(K²·N) 会炸 |
| `run_metrics` | `True` | scib 整合评分 |
| `run_rogue` | `True` | 每簇 ROGUE 纯度(需要 `layers['counts']` 或 `adata.raw` 存原始 counts)|
| `use_anndataoom` | `False` | 1M+ 细胞 densify 防 OOM,先 `pip install anndataoom` |
| `pca_n_comps` | `"auto"` | Gavish-Donoho 硬阈值自动选;也可给整数固定 |

### 典型用法模板

```python
# A. 小数据严格验证(≤ 10k,5 tech)
run_pipeline(h5, batch_key="tech", run_harmony=True, run_bbknn=True,
             run_recall=True, run_rogue=True)

# B. 单样本 SCOP standard_scop 风格(1k-5k,无 batch)
run_pipeline(h5, batch_key="sample", run_harmony=False, run_bbknn=False,
             run_recall=True, run_rogue=True)

# C. 大数据 atlas(100k+,多数据集)
run_pipeline(h5, batch_key="dataset", run_harmony=True, run_bbknn=True,
             run_recall=False, run_rogue=True, use_anndataoom=True)
```

---

## 已支持的整合方法(v0.1 当前)

| 方法 | 范式 | 实现 | scib 排名(Luecken 2022)|
|---|---|---|---|
| **Harmony 2.0** | 线性 embedding(soft k-means + MoE ridge)| ✅ Rust 原生(scatlas)| #2-3 综合 |
| **BBKNN** | 图论 batch-balanced kNN | ✅ Rust 原生(scatlas)| #5(批次去除强)|
| **Uncorrected** | 无校正 baseline | 直接走 PCA | baseline |

---

## 🚧 下一步路线图(整合方法扩展愿景)

下面是**还没实现**的方法,按原理范式 + scib benchmark 排名规划。目标是打造一个 **7 方法范式覆盖** 的严谨对比平台。

### 📋 v0.2 计划(Rust 加速优先级 P0)

| 方法 | 范式 | Rust 化难度 | 优先级 | 备注 |
|---|---|---|---|---|
| **scVI** | 深度 VAE(ZINB)| ❌ 不重写(PyTorch 已够快,GPU 必备)| P0 | **scib #1**,必须接入 benchmark |
| **Scanorama** | 全对 MNN + SVD | ⚠️ 可 Rust 化(SVD + kNN 已有) | P0 | pair-wise MNN 范式代表 |
| **fastMNN** | 线性 batchelor | ⚠️ 可 Rust 化(batchelor 算法相对简单) | P0 | Bioconductor 生态基线 |

**v0.2 目标**:新增上述 3 种,配合 Harmony + BBKNN + Uncorrected → 6 方法 benchmark panel,覆盖 scib 前 5 名。

### 📋 v0.3 计划(Rust 加速优先级 P1)

| 方法 | 范式 | Rust 化难度 | 优先级 | 备注 |
|---|---|---|---|---|
| **scANVI** | 半监督 VAE(scVI + label)| ❌ scvi-tools 胶水 | P1 | **scib #1 并列**,要有 coarse label 数据集 |
| **LIGER (iNMF)** | 整合 NMF | ✅ **适合 Rust 化**(NMF 我们已有 fast_3ca 经验)| P1 | 非 deep 非线性唯一代表;Rust 化预期 10× |
| **Symphony** | Harmony reference mapping | ⚠️ 可 Rust 化(基于我们已有 Harmony)| P1 | atlas 查询标准 |

### 📋 v0.4+ 远期愿景

- **CSS / Coralysis**:小众 embedding 方法,覆盖度补全
- **scArches / scPoli**:reference mapping GPU 路径
- **GLUE / Multigrate**:多模态(RNA + ATAC)
- **GPU 加速 scVI 路径**:PyTorch native,预期 5-20× vs CPU

### ❌ 显式排除(已判定不做)

| 方法 | 跳过原因 |
|---|---|
| MNN 经典 | O(n²) 过慢,被 fastMNN 完全取代 |
| Conos | scib benchmark 垫底,图范式已被 BBKNN 代表 |
| ComBat | 过度校正损伤生物信号 |
| Seurat v4 CCA | 被 Seurat v5 RPCA 取代 |
| scSHC | 历史稳定性差 + Rust 生态缺 ARPACK/Ward/Cholesky |

### 📊 同时规划的 scib-metrics 补齐(v0.2 一起做)

当前我们的 scib 指标是 **5 个**(iLISI / cLISI / graph_connectivity / kBET / silhouette)。Luecken 2022 完整集是 **10 个**,还差:

- NMI(Normalized Mutual Information)— 聚类-标签相似度
- ARI(Adjusted Rand Index)
- isolated label F1 — 稀有 celltype 保真度
- PCR(Principal Component Regression)— batch 对 PCA 的解释力
- cell cycle conservation — 生物信号保真度(可选)

完整计划见 [ROADMAP.md](ROADMAP.md)。

---

### Benchmarks

```bash
# panc8 跨 tech 整合对比(3-way: baseline / SCOP / scatlas)
python scatlas_pipeline/benchmarks/compare_panc8_full.py

# pancreas 单样本 SCOP 对比
python scatlas_pipeline/benchmarks/compare_pancreas_sub.py

# 157k 全流程 demo(需要自备 epithelia_full.h5ad)
python scatlas_pipeline/benchmarks/run_157k.py
```

---

## 依赖清单

### 系统
- **Linux x86_64** 或 **WSL2 Ubuntu 20.04+**(Windows 原生未测试)
- 16+ cores 推荐(rayon 多线程)
- **Rust ≥ 1.75**(建议 rustup 最新 stable)
- **Python ≥ 3.10**

### Python 运行时
- numpy ≥ 1.26
- scipy ≥ 1.11
- anndata ≥ 0.10
- pandas ≥ 2.0
- scikit-learn ≥ 1.3
- scanpy ≥ 1.11
- leidenalg + python-igraph
- matplotlib(benchmark 画图)
- rdata(可选,读 Seurat .rda 文件)
- scvi-tools(可选,scVI 整合对比)

### 构建时
- maturin ≥ 1.7
- pyo3 0.24
- ndarray 0.16 + rayon 1.10

完整锁定见 `scatlas/Cargo.toml` / `scvalidate_rewrite/scvalidate_rust/Cargo.toml`。

---

## 目录结构

```
fast_auto_scRNA/
├── README.md                    # 本文件
├── INSTALL.md                   # 安装详细步骤
├── UPDATE.md                    # 如何升级 / 开发新版本
├── LICENSE                      # MIT
├── setup.sh                     # 一键安装脚本
├── .gitignore
│
├── scatlas/                     # Rust 核心 #1
│   ├── Cargo.toml               # workspace
│   ├── crates/
│   │   ├── scatlas-core/        # pure Rust kernels
│   │   └── scatlas-py/          # PyO3 bindings
│   ├── python/tests/            # 48 pytest parity tests
│   └── benchmark/               # m4_harmony.py / m5_pca.py / m7_umap.py / full_pipeline.py
│
├── scvalidate_rewrite/          # Rust 核心 #2
│   ├── scvalidate/              # Python 编排层
│   ├── scvalidate_rust/         # Rust kernels (wilcoxon, knockoff)
│   └── benchmark/
│
└── scatlas_pipeline/            # 一键编排层
    ├── pipeline.py              # run_pipeline / PipelineConfig
    ├── configs/                 # YAML 示例
    └── benchmarks/
        ├── run_157k.py
        ├── compare_panc8_full.py
        └── compare_pancreas_sub.py
```

---

## 关键参考文献

- **PCA 随机 SVD**: Halko, Martinsson, Tropp 2011
- **Gavish-Donoho 硬阈值**: Gavish & Donoho 2014, IEEE TIT
- **Harmony 2.0**: Korsunsky et al. 2019, Nature Methods;pati-ni/harmony C++ 源码
- **BBKNN**: Polański et al. 2019, Bioinformatics
- **UMAP**: McInnes et al. 2018;umap-learn 实现
- **scib-metrics**: Luecken et al. 2022, Nature Methods
- **scValidate (recall)**: knockoff-filtered 差异表达验证聚类

---

## License

MIT — 见 [LICENSE](LICENSE)。

---

## 维护与更新

见 [UPDATE.md](UPDATE.md)。

发 issue / PR 欢迎。

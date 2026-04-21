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

## 整合方法选型建议(7 种代表)

SCOP 支持 15 种整合方法,为避免过度分散,**按原理范式选 7 个最有代表性的**(Luecken 2022 scib 排名 + 最广使用):

| 范式 | 推荐方法 | 是否 Rust 原生 | 备注 |
|---|---|---|---|
| **线性 embedding(soft k-means)** | **Harmony 2.0** | ✅ scatlas | scib #2-3,最广用 |
| **线性 embedding(batchelor)** | fastMNN | ❌(R)| 与 Harmony 范式差异,Bioconductor 基线 |
| **深度 VAE** | scVI | ❌(PyTorch)| **scib #1**,GPU 加速强 |
| **图论 batch-balanced** | **BBKNN** | ✅ scatlas | 批次去除强 |
| **全对 MNN** | Scanorama | ❌(Python)| pair-wise,与 BBKNN 范式互补 |
| **iNMF** | LIGER | ❌(R)| 非 deep 非线性唯一代表 |
| **Baseline** | Uncorrected | — | 无校正参照 |

**跳过的方法**(按范式已被代表覆盖或已过时):
- MNN 经典(O(n²) 过慢)/ Conos(scib 差)/ ComBat(过度校正)/ Seurat v4 CCA(被 v5 取代)

完整对比协议见 [ROADMAP.md](ROADMAP.md) — v0.2 计划加 scVI/Scanorama/fastMNN 三种,v0.3 再加 LIGER/scANVI/Symphony。

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

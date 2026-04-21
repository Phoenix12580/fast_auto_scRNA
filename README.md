# fast_auto_scRNA

**一键自动化 scRNA-seq 分析流程 — Rust 加速,Python API**

端到端覆盖 `load → QC → 归一化 → PCA → BBKNN → Harmony → UMAP → Leiden → recall-validated 聚类 → scib metrics`,所有热路径用 Rust 重写,同等质量下比纯 Python 流程快 **20-45×**。

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
    run_harmony=True,          # 跨 batch 时开
    run_recall=True,           # knockoff 验证聚类(小中等数据集)
    out_h5ad="atlas.h5ad",
)
print(f"clusters: {adata.obs['leiden'].nunique()}")
print(f"scib mean: {adata.uns['scib_score']['mean']:.3f}")
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

# fast_auto_scRNA v2

端到端单细胞 RNA-seq 图谱分析管线 —— Rust 加速的核心内核 + Python 编排层，
**按管线阶段组织**（不按历史库划分）。每个阶段都是自包含的模块。

---

## 🟢 状态 (2026-04-26，分支 `main`)

**当前里程碑：v2-P12 完成**（CHAMP resolution picker 上线）

最近 milestone 摘要（详细见 [ROADMAP.md](ROADMAP.md)）：

| 版本 | 内容 | 收益 |
|---|---|---|
| **v2-P12** (`74e8dfd`) | **CHAMP** (Weir 2017) 替掉 knee/conductance 启发式；Phase 2b 跳过重算 scIB | **−32 min on 222k**（100→68 min） |
| v2-P11 (`641708f`) | Phase 2a scIB 多进程脚手架（默认关，BLAS-bound 验证后 0.92×） | 0（但留作将来 GPU pivot 入口） |
| v2-P10 (`5ac6075`) | Cancer Cell 2024 风格 scIB 全套 + 4 路集成（bbknn/harmony/fastmnn/scvi） | 论文级指标面板 |
| v2-P9.x | Phase 2 拆分，winner-only Leiden + 多核 worker priority | 30 min 省 |
| GS-5 | Conductance-based picker（已被 v2-P12 CHAMP 替代）| 历史 |
| GS-4 | Parallel Leiden sweep via ProcessPoolExecutor | 3.5× |
| GS-3 (`5ba6d83`) | Rust `silhouette_precomputed` 内核（40× vs sklearn）| 仍在用 |

**当前 222k 全管线性能**（StepF prostate atlas，16-core WSL）：
- 默认 `integration="all"`（3 路：bbknn / harmony / fastmnn）：**~46 min**
- 4 路（`integration="all+scvi"`，含 GPU scVI 22 min）：**~68 min**

**核心设计**
- **WSL is the default dev environment**（除 scVI/CUDA torch 外的一切）—— 见 [INSTALL.md](INSTALL.md)
- **多路集成对比**：默认 3 路（bbknn / harmony / fastmnn），按 scIB mean 自动选 winner（带 human-decision gate）；scvi 显式 opt-in（`integration="all+scvi"`），原因是 atlas 上 22 min wall driver + 需 CUDA torch
- **CHAMP 分辨率选择**：Weir 2017 凸包模块度法，确定性 + 数学坚实，~30 leidens vs 旧 knee 的 150
- **scIB 论文级指标**：iLISI / cLISI / graph_connectivity / 3×silhouette (label/batch/isolated) / ROGUE / SCCAF
- **Cache-first**：所有中间产物（labels / curves / embeddings）都写回 h5ad，二次访问免重算

---

## 快速开始

### WSL（推荐）

```bash
# 1) WSL Ubuntu 24.04，从仓库根目录开始
cd /mnt/f/fast_auto_scRNA_v2
uv venv .venv-wsl --python 3.12
source .venv-wsl/bin/activate

# 2) 装系统依赖（Rust + BLAS）
sudo apt install build-essential libopenblas-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 3) Rust kernel + Python wheel
uv pip install pip maturin
CARGO_TARGET_DIR=$HOME/cargo-target/fast_auto_scrna \
  maturin develop --release -m rust/Cargo.toml

uv pip install -e .

# 4) 跑测试（13 passed, 4 scvi-skipped on WSL — scvi 默认不装）
pytest tests/ -v
```

### Windows（仅当需要 scVI/CUDA torch 时）

```powershell
cd F:\fast_auto_scRNA_v2
uv venv --python 3.12
.venv\Scripts\activate
maturin develop --release -m rust\Cargo.toml
uv pip install -e ".[scvi]"
# CUDA torch（scvi-tools 默认拉 CPU torch，必须强制重装）：
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu121 torch
pytest tests/ -v
```

完整安装步骤见 [INSTALL.md](INSTALL.md)。

---

## 目录布局 —— 按管线阶段组织

| # | 模块 | 职责 | 加速 |
|---|------|------|---|
| 01 | `fast_auto_scrna/io/` | 加载 h5ad / rda / Seurat-qs；逐 cell QC | Python |
| 02 | `preprocess/normalize.py` | normalize_total + log1p | Python |
| 03 | `preprocess/hvg.py` | 高变基因（seurat_v3 VST on counts）| Python (scikit-misc) |
| 04 | `preprocess/scale.py` | z-score + max 截断 | Python |
| 05 | `pca/` | 随机化 PCA + Gavish-Donoho 自动 n_comps | **Rust** |
| 06 | `integration/` | bbknn / harmony / fastmnn（默认 3 路）+ scvi（opt-in via `"all+scvi"`）| **Rust** (bbknn, harmony) / Python (fastmnn) / GPU torch (scvi) |
| 07 | `neighbors/` | kNN + fuzzy_simplicial_set | **Rust** |
| 08 | `scib_metrics/` | iLISI / cLISI / graph_conn / 3×silhouette / kBET (opt-in) | **Rust** (LISI/GC/kBET) + JAX (scib-metrics ASW) |
| 09 | `umap/` | UMAP layout SGD | **Rust** |
| 10 | `cluster/` | **Leiden + CHAMP picker (Weir 2017)** | leidenalg + Python |
| 11 | `rogue/` | 按簇纯度（entropy + loess 拟合）| **Rust** |
| — | `scib_metrics/sccaf.py` | SCCAF accuracy | sklearn |
| — | `config.py` | `PipelineConfig` —— 所有参数集中一处 | — |
| — | `runner.py` | `run_from_config(cfg, adata_in=None)` 主入口 + Phase 1/2a/2b/2c 编排 + human-decision gate | — |
| — | `_native/` | Rust PyO3 绑定的薄 re-export | maturin / abi3-py310 |

---

## Rust workspace

```
rust/
├── Cargo.toml                            workspace 根
└── crates/
    ├── kernels/                          纯 Rust 算法内核（rlib，无 PyO3）
    │   └── src/
    │       ├── pca.rs                    randomized SVD + 3 种 elbow 选择器
    │       ├── bbknn.rs                  batch-balanced kNN
    │       ├── harmony/                  Harmony 2 (k-means + correct steps)
    │       ├── umap.rs                   UMAP layout SGD
    │       ├── fuzzy.rs                  fuzzy_simplicial_set
    │       ├── metrics/                  lisi / graph_conn / kbet
    │       ├── rogue.rs                  entropy_table + calculate_rogue
    │       └── silhouette.rs             precomputed-distance silhouette (GS-3，40× vs sklearn)
    └── py_bindings/                      PyO3 → fast_auto_scrna._native
```

---

## CHAMP resolution picker（v2-P12 默认）

实现 Weir, Emmons, Wakefield, Hopkins, Mucha (2017) "Post-processing partitions
to identify domains of modularity optimization" (*Algorithms* 10(3):93)。

**核心数学**：对任何固定 partition P，Newman 模块度 Q 是 γ 的线性函数：

$$Q(\gamma; P) = a_P - \gamma \cdot b_P$$

每个候选 partition 是 (γ, Q) 平面上的一条直线。所有候选直线的**上包络线**
对应于 (b, a) 点集的**上凸包**。CHAMP 算法：

1. 在 30 个 γ 值上跑 Leiden，去重
2. 对每个 unique partition 算 (a, b)
3. 找上凸包（Andrew 单调链算法）
4. 每个 hull partition 的「admissible γ-range」= 被相邻 crossover γ 夹起来的区间
5. 选 admissible γ-range 最宽的（log-width 度量，因为 γ-空间 scale-free）

**vs 旧 knee picker**：CHAMP 是确定性 + 几何，无脆弱启发式 detector。
222k bbknn 上：knee 跑 150 leidens 选 k=12，CHAMP 跑 30 leidens 选 k=8
（与 ct.sub GT k=7 更对齐），加速 5.18×。

可视化：`benchmarks/out/champ_landscape_222k.png`（pipeline 默认输出 `champ_curve_<method>.png`）

---

## 测试数据

- `data/pancreas_sub.rda` —— 1000 cell 胰腺，1 batch
  （软链 → `F:/NMF_rewrite/pancreas_sub.rda`）。单元测试基准。
- `data/StepF.All_Cells.h5ad` —— 222 529 cells × 20 055 genes 前列腺图谱，
  10 batches，ct.main（3 类）/ ct.sub（7 类）/ ct.sub.epi（13 类）GT 标签
  （软链 → `F:/NMF_rewrite/StepF.All_Cells.h5ad`）。图谱尺度基准。

## v2 明确**不包含**的内容

- **`recall`** —— scvalidate 的 recall 簇数选择器全弃。簇数选择由
  `cluster/champ.py` 的 CHAMP picker 接管。
- **`wilcoxon` / `knockoff` Rust 内核** —— 只服务于 recall，一并丢弃。
- **knee / conductance / silhouette / target_n optimizers**（v2-P12 删除）——
  全部启发式 picker 都被 CHAMP 替掉。原代码留在 git 历史里（commit `711223f` 之前）。

## 历史沿革

v2 从 v1 提交 `c1107e8` 切出。v1 存活于 `F:/NMF_rewrite/fast_auto_scRNA_v1/`
（分支 `v1`），已归档。所有新工作在这里进行。

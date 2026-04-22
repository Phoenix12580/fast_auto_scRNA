# INSTALL.md — 安装指南

## 系统要求

| | 版本 |
|---|---|
| 操作系统 | Linux x86_64 / WSL2 Ubuntu 20.04+(macOS 未测试,Windows 原生不支持) |
| CPU | 推荐 16+ cores(Rust 并行加速主要靠此) |
| RAM | 1k 细胞:4GB;157k:16GB;1M:64GB 推荐 |
| Rust | ≥ 1.75 stable |
| Python | ≥ 3.10 |

## 方式 A:一键脚本(推荐)

```bash
git clone https://github.com/Phoenix12580/fast_auto_scRNA.git
cd fast_auto_scRNA
./setup.sh
```

脚本会:
1. 检查 Rust + Python + uv(没装会提示怎么装)
2. 创建 `.venv/` 并装 Python 依赖
3. `maturin develop --release` 编译两个 Rust 扩展
4. 跑 smoke test 验证

成功后 activate:
```bash
source .venv/bin/activate
python -c "from scatlas_pipeline import run_pipeline; print('OK')"
```

## 方式 B:手动步骤(排错或定制时用)

### 1. 装 Rust 工具链

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version   # 应 ≥ 1.75
```

### 2. 装 uv (Python 包管理器,替代 pip/venv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # 或重开 shell
```

### 3. 创建虚拟环境

```bash
cd fast_auto_scRNA
uv venv --python 3.10
source .venv/bin/activate
```

### 4. 装 Python 依赖

```bash
uv pip install \
    maturin>=1.7 \
    numpy scipy pandas \
    anndata scanpy \
    scikit-learn matplotlib \
    leidenalg python-igraph \
    umap-learn \
    pytest
```

可选(读 Seurat `.rda` 文件用):
```bash
uv pip install rdata
```

### 5. 编译 scatlas(Rust kernels #1)

```bash
cd scatlas/crates/scatlas-py
maturin develop --release
cd ../../..
```

### 6. 编译 scvalidate_rust(Rust kernels #2)

```bash
cd scvalidate_rewrite/scvalidate_rust
maturin develop --release
cd ../..
```

### 7. 装 scvalidate Python 包

```bash
uv pip install -e scvalidate_rewrite
```

### 8. 装 scatlas_pipeline

```bash
uv pip install -e scatlas_pipeline
```

(或直接把 `scatlas_pipeline/` 放 PYTHONPATH)

### 9. 验证

```bash
# 单元测试
cd scatlas && cargo test --release -p scatlas-core
pytest python/tests/

# Smoke test
python -c "
from scatlas_pipeline import run_pipeline
import numpy as np, scipy.sparse as sp, anndata as ad, tempfile, os
X = sp.csr_matrix(np.random.default_rng(0).poisson(1, (200, 100)).astype(np.float32))
a = ad.AnnData(X=X)
a.obs['batch'] = np.where(np.arange(200) < 100, 'A', 'B')
with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as f:
    a.write_h5ad(f.name)
    try:
        out = run_pipeline(f.name, batch_key='batch', min_genes=5, min_cells=1)
        print(f'OK — {out.n_obs} cells, {out.obs[\"leiden\"].nunique()} clusters')
    finally:
        os.unlink(f.name)
"
```

## 排错

### `error[E0277]` 之类 Rust 编译错误
- 确认 rustc ≥ 1.75:`rustc --version`
- 清理重试:`cargo clean && maturin develop --release`

### `ModuleNotFoundError: scvalidate_rust`
- 必须在 `scvalidate_rust/` 目录运行 maturin,且在**目标 venv 激活**时运行
- 检查:`python -c "import scvalidate_rust; print(scvalidate_rust.__file__)"`

### Python wheel 装在错 venv
- maturin 会根据 `python` 找 venv。确保:
  ```bash
  which python   # 应指向 .venv/bin/python
  ```
- 或显式指定:`maturin develop --release --python .venv/bin/python`

### BBKNN 连接矩阵跑到 umap-learn 而不是 Rust
- scatlas 的 Rust `fuzzy_simplicial_set` 需要 `scatlas._scatlas_native.ext.fuzzy_simplicial_set` 可导入
- 检查:
  ```python
  from scatlas._scatlas_native.ext import fuzzy_simplicial_set
  ```
  如果报错,重装 scatlas:`cd scatlas/crates/scatlas-py && maturin develop --release`

### 内存不足(≥30k cells)
- v1 起 recall 自动走 **anndata-oom backend**,augmented 矩阵写 scratch 磁盘;≥30k 时触发,157k 峰值约 2.5-7 GB。详见下方"anndataoom"节。
- PCA auto 模式跑 60 comps,大数据切显式 30:`pca_n_comps=30`

## anndataoom (≥30k cells, strongly recommended on WSL2/Linux)

`recall` 在 ≥30k cells 时自动走 anndata-oom backend,把 augmented 矩阵写 scratch 磁盘,解决 157k 级别的 OOM。

Linux / macOS(推荐,有 wheel):
```bash
pip install anndataoom
# or as a scvalidate optional extra:
pip install -e "scvalidate_rewrite[oom]"
```

Windows:没有预编译 wheel,需要 MSVC + vcvarsall + pip cmake 走 HDF5 源码编译(10-20 分钟)。推荐 WSL2。

---

## 平台说明

### WSL2(当前主力)
- 所有数据放 Windows `F:\` 盘,WSL 通过 `/mnt/f/` 访问
- 速度:drvfs 比 WSL 原生慢 3-5×,但对 scRNA 单次读取不明显
- 多核:WSL2 会继承 Windows CPU,`nproc` 看到的是物理核数

### 纯 Linux
- 推荐,编译和运行速度都更好
- 大数据(1M+)建议 Linux 原生而非 WSL

### macOS(Apple Silicon)
- 理论可行但未测试
- 需要 `rustup target add aarch64-apple-darwin`
- `matrixmultiply` / `hnsw_rs` / `rayon` 都支持 M 系列
- matplotlib 可能要装 `python-tk` 依赖

### Windows 原生
- **不支持**。Rust 侧能编,但 scipy/scanpy 的某些 C 扩展在 Windows 下装麻烦
- 一律用 WSL2

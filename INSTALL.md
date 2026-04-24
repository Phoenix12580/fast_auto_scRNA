# 安装 —— fast_auto_scRNA v2

## 依赖要求

| 工具 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 3.10.20 已测 |
| Rust | 1.95+ | stable 工具链 + `clippy` + `rustfmt` |
| uv | 0.11+ | venv + pip 的替代品 |
| maturin | 1.13+ | PyO3 → wheel 构建器 |
| WSL2（Windows） | — | 做任何正经计算都建议走 WSL2 |

Python 依赖（自动装上）：`anndata`、`scanpy`、`scipy`、`numpy`、`scikit-learn`、`pandas`、`igraph`、`leidenalg`、`umap-learn`、`hnswlib`、`rdata`、`h5py`、`pytest`。

## 全流程 —— 从零搭建

### 1. 克隆 + 进入工作树

```bash
git clone https://github.com/Phoenix12580/fast_auto_scRNA.git
cd fast_auto_scRNA
# 如果你已有裸仓并想挂一个新的 worktree：
git worktree add ../fast_auto_scRNA_v2
cd ../fast_auto_scRNA_v2
```

### 2. Python 虚拟环境

```bash
uv venv --python 3.10
source .venv/Scripts/activate   # Windows Git Bash
# 或：source .venv/bin/activate # WSL / Linux
```

### 3. 编译并安装 Rust 绑定

```bash
uv pip install maturin
maturin develop --release -m rust/crates/py_bindings/Cargo.toml
```

这会编译 `kernels` + `py_bindings` 并把产物 wheel 装进 `.venv`。编译出的模块以
`fast_auto_scrna._native` 的形式被 import。

### 4. 安装 Python 包（editable）

```bash
uv pip install -e .
```

### 5. 链接测试数据

```bash
# WSL / Linux：
ln -s /mnt/f/NMF_rewrite/pancreas_sub.rda     data/pancreas_sub.rda
ln -s /mnt/f/NMF_rewrite/StepF.All_Cells.h5ad data/StepF.All_Cells.h5ad

# Windows（管理员）：
mklink data\pancreas_sub.rda     F:\NMF_rewrite\pancreas_sub.rda
mklink data\StepF.All_Cells.h5ad F:\NMF_rewrite\StepF.All_Cells.h5ad
```

### 6. 验证

```bash
pytest tests/ -v
python -c "import fast_auto_scrna; print(fast_auto_scrna.__version__)"
```

## 仅 Rust 侧工作流（不走 Python 也能改内核）

```bash
cargo check --manifest-path rust/Cargo.toml --all
cargo test  --manifest-path rust/Cargo.toml --release
cargo clippy --manifest-path rust/Cargo.toml --all -- -D warnings
```

## Rust 改完之后重建

```bash
maturin develop --release -m rust/crates/py_bindings/Cargo.toml
```

（够快 —— 增量编译只重建动过的内核。）

## Windows + WSL 注意事项

- Git 操作必须在 Windows shell（Git Bash / PowerShell）里跑。`.git`
  文件里是 Windows 路径，WSL 的 git 解析不了。
- Python / Rust / maturin 两边都能跑，但大数据建议走 WSL：通过 `/mnt/f/`
  访问 NTFS 比原生 ext4 慢很多。
- 两种环境下 `.venv/` 的激活脚本路径不同，按第 2 步选对应的那行。

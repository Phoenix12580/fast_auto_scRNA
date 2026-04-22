# Install — fast_auto_scRNA v2

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | 3.10.20 tested |
| Rust | 1.95+ | stable toolchain + `clippy` + `rustfmt` |
| uv | 0.11+ | venv + pip replacement |
| maturin | 1.13+ | PyO3 → wheel builder |
| WSL2 (Windows) | — | recommended for any real compute |

Python deps (auto-installed): `anndata`, `scanpy`, `scipy`, `numpy`, `scikit-learn`, `pandas`, `igraph`, `leidenalg`, `umap-learn`, `hnswlib`, `rdata`, `h5py`, `pytest`.

## Full setup — from scratch

### 1. Clone + enter worktree

```bash
git clone https://github.com/Phoenix12580/fast_auto_scRNA.git
cd fast_auto_scRNA
git checkout v2
# Or if you already have the bare repo + want a new worktree:
git worktree add ../fast_auto_scRNA_v2 v2
cd ../fast_auto_scRNA_v2
```

### 2. Python environment

```bash
uv venv --python 3.10
source .venv/Scripts/activate   # Windows Git Bash
# or:  source .venv/bin/activate # WSL / Linux
```

### 3. Build + install the Rust bindings

```bash
uv pip install maturin
maturin develop --release -m rust/crates/py_bindings/Cargo.toml
```

This compiles `kernels` + `py_bindings` and installs the resulting wheel into
`.venv`. The compiled module is importable as `fast_auto_scrna._native`.

### 4. Install the Python package (editable)

```bash
uv pip install -e .
```

### 5. Link the test data

```bash
# WSL / Linux:
ln -s /mnt/f/NMF_rewrite/pancreas_sub.rda     data/pancreas_sub.rda
ln -s /mnt/f/NMF_rewrite/StepF.All_Cells.h5ad data/StepF.All_Cells.h5ad

# Windows (as admin):
mklink data\pancreas_sub.rda     F:\NMF_rewrite\pancreas_sub.rda
mklink data\StepF.All_Cells.h5ad F:\NMF_rewrite\StepF.All_Cells.h5ad
```

### 6. Verify

```bash
pytest tests/ -v
python -c "import fast_auto_scrna; print(fast_auto_scrna.__version__)"
```

## Rust-only workflow (rebuild kernel without Python)

```bash
cargo check --manifest-path rust/Cargo.toml --all
cargo test  --manifest-path rust/Cargo.toml --release
cargo clippy --manifest-path rust/Cargo.toml --all -- -D warnings
```

## Rebuild after Rust changes

```bash
maturin develop --release -m rust/crates/py_bindings/Cargo.toml
```

(fast enough — incremental compilation only rebuilds touched kernels.)

## Windows + WSL notes

- Git operations must run in a Windows shell (Git Bash / PowerShell). The
  `.git` file uses Windows paths; WSL's git can't resolve them.
- Python / Rust / maturin all run fine in either. WSL is faster for large
  datasets because NTFS I/O via `/mnt/f/` is slower than ext4.
- The `.venv/` activation script differs between environments; use the
  matching one from step 2.

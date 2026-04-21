# UPDATE.md — 升级和开发新版本

## 用户:升级已安装环境

```bash
cd fast_auto_scRNA
git pull

# 如果 Rust 代码有改动,重编译两个扩展
source .venv/bin/activate
cd scatlas/crates/scatlas-py && maturin develop --release && cd ../../..
cd scvalidate_rewrite/scvalidate_rust && maturin develop --release && cd ../..

# 如果 Python 代码或 pyproject.toml 有改动
uv pip install -e scvalidate_rewrite
uv pip install -e scatlas_pipeline
```

一行完成:
```bash
git pull && ./setup.sh --update
```

## 开发者:发布新版本的流程

### 版本号约定

语义化版本 **major.minor.patch**:
- `patch`(0.1.0 → 0.1.1):bug 修复,算法无改动,不破坏 API
- `minor`(0.1.0 → 0.2.0):新增功能,向后兼容
- `major`(0.1.0 → 1.0.0):API 破坏性改动

三个子包各自独立版本,但在 tag 时绑定主仓库版本:
- `scatlas/crates/scatlas-core/Cargo.toml` 里 `version.workspace = true`
- `scvalidate_rewrite/scvalidate_rust/Cargo.toml` 里 `version = "0.1.0"`
- `scatlas_pipeline` 暂无 pyproject.toml,跟主仓库 tag

### 开发分支流程

```bash
# 1. 从 main 开新分支
git checkout -b feat/my-new-kernel

# 2. 写代码 / 改文档 / 加测试
# 确保通过:
cd scatlas && cargo test --release && cargo clippy --release --all -- -D warnings
cd ../scvalidate_rewrite/scvalidate_rust && cargo test --release
cd ../.. && source .venv/bin/activate && pytest scatlas/python/tests/

# 3. 先本地冒烟测
python scatlas_pipeline/benchmarks/compare_pancreas_sub.py

# 4. commit + push
git add -A
git commit -m "feat: add X kernel with Y% speedup"
git push -u origin feat/my-new-kernel

# 5. 开 PR 到 main
```

### 发正式版

```bash
# 1. 切回 main 拉最新
git checkout main && git pull

# 2. 改版本号
# - scatlas/Cargo.toml         [workspace.package] version = "0.2.0"
# - scvalidate_rewrite/scvalidate_rust/Cargo.toml  version = "0.2.0"
# (scatlas_pipeline 无独立版本号)

# 3. 更新 CHANGELOG.md(如果有)

# 4. commit + tag + push
git add -A
git commit -m "chore: release v0.2.0"
git tag -a v0.2.0 -m "v0.2.0 — new features X, Y, Z"
git push origin main --tags
```

### 发预编译 wheel(可选,未来扩展)

目前用户装需要本地编译 Rust(5-10 min)。想发 wheel 到 PyPI:

```bash
# 装 maturin + cibuildwheel
uv pip install maturin cibuildwheel

# 在 GitHub Actions 里跑(见 .github/workflows/wheels.yml,待补)
# 构建 manylinux_2_34 x86_64 wheel,上传 PyPI
cd scatlas/crates/scatlas-py
maturin publish --release --skip-existing

cd ../../../scvalidate_rewrite/scvalidate_rust
maturin publish --release --skip-existing
```

## 仓库架构注意

### 为啥三个子包放在一起

历史上 `scatlas` 和 `scvalidate_rewrite` 是独立项目,在开发期间 `scvalidate_rewrite/scvalidate_rust` 的 stats kernels(wilcoxon / knockoff)被迁入 `scatlas/crates/scatlas-core/src/stats/` 重用。目前两边代码基本同源,但独立编译发布。

以后可以选:
- **合并**:把 scvalidate_rust 删掉,只保留 scatlas 的 stats 模块,Python 编排层(scvalidate)直接调 scatlas
- **拆分**:把每个子包放独立 GitHub 仓库,用 submodule/subtree 引入

### 测试覆盖

- `scatlas/python/tests/`:48 pytest parity 测试(vs scipy/sklearn/umap-learn)
- `scatlas/crates/scatlas-core`:57 Rust 单元测试
- `scvalidate_rewrite/scvalidate_rust`:3 Rust 单元测试
- `scvalidate_rewrite/tests/`:Python parity 测试(vs R 的 scValidate)

发版前必须全部通过 + clippy 无警告。

### Benchmark 基线

- `scatlas/benchmark/full_pipeline.py` — 157k epithelia,3.3 min 目标
- `scatlas_pipeline/benchmarks/compare_panc8_full.py` — 30× vs SCOP 目标
- `scatlas_pipeline/benchmarks/compare_pancreas_sub.py` — 20× vs SCOP-standard 目标

退步 > 10% 视为 regression,发布前需调查。

## 内部 ABI 变动记录

| 版本 | 日期 | API 破坏 | 说明 |
|---|---|---|---|
| 0.1.0 | 2026-04-21 | — | 初版 |

## 联系

PR / Issue: https://github.com/Phoenix12580/fast_auto_scRNA

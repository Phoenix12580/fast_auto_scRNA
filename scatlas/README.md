# scatlas

Rust-first scRNA-seq atlas computation package.

Workspace layout:

- `crates/scatlas-core` — pure Rust kernels (preprocessing, integration, metrics, stats)
- `crates/scatlas-py` — PyO3 bindings, published as `pip install scatlas`
- `python/tests` — parity tests against scanpy / scib-metrics / R harmony
- `benchmark` — performance scripts

See `plans/happy-sparking-falcon.md` for the full roadmap (M1 → M7).

## Dev environment

All builds and tests run inside WSL2. Source lives on Windows (`F:\NMF_rewrite\scatlas\`)
and is accessed from WSL as `/mnt/f/NMF_rewrite/scatlas/` via drvfs.

```bash
# first time
uv venv
uv pip install maturin numpy scipy pytest

# build + install
maturin build --release --manifest-path crates/scatlas-py/Cargo.toml
uv pip install --force-reinstall --no-deps target/wheels/scatlas-*.whl

# test
cargo test --workspace
pytest python/tests/
```

# fast_auto_scRNA v2

End-to-end single-cell RNA-seq atlas pipeline — Rust-accelerated kernels with a
Python orchestration layer, organized **by pipeline stage** (not by legacy
library). Each stage is a self-contained module.

---

## 🟢 Status (2026-04-24, branch `main`)

**Done**
- v2 worktree at `F:/fast_auto_scRNA_v2`, branch `main` (renamed from `v2`; old main preserved as tag `legacy-main-2026-04-22`)
- Branched from v1 commit `c1107e8` (includes GS-2 graph-silhouette pipeline wiring + validated 222k atlas smoke)
- v1 legacy tree (scatlas/, scatlas_pipeline/, scvalidate_rewrite/, scripts/, docs/images/, docs/superpowers/) removed — **166 files, 26 426 lines deleted** in commit `32787b2`
- New stage-organized skeleton: `fast_auto_scrna/{io, preprocess, pca, integration, neighbors, scib_metrics, umap, cluster, rogue, common, _native}` + `rust/crates/{kernels, py_bindings}` + empty `tests/`, `benchmarks/`, `docs/{specs,plans}/`
- **V2-P1** — Rust kernels + PyO3 bindings migrated from v1 `scatlas-core` / `scatlas-py` (commit `91e2aef`):
  - `rust/crates/kernels/src/`: 7 modules — `bbknn`, `fuzzy`, `harmony/`, `metrics/`, `pca`, `rogue` (promoted out of v1 `stats/`), `umap`
  - `rust/crates/py_bindings/src/`: adapters re-registered under v2 stage names — `_native.{pca, bbknn, harmony, fuzzy, metrics, umap, rogue}`; BBKNN promoted out of v1 `ext/` grab bag
  - Dropped recall-only: `stats/wilcoxon.rs` (225 LOC) + `stats/knockoff.rs` (74 LOC) + their PyO3 wrappers
  - `cargo check --workspace` + `cargo clippy --workspace --all-targets`: green
- Test data symlinked in `data/`: `pancreas_sub.rda` (3.5 MB, 1-batch pancreas) + `StepF.All_Cells.h5ad` (1.66 GB, 222 k cells × 20 055 genes, 10 batches)

**Next (in order)**
1. **V2-P2** — carve `scatlas_pipeline/pipeline.py` (1 140 LOC) into `fast_auto_scrna/` stage modules. `PipelineConfig` → `config.py`, `run_from_config` → `runner.py`. Bring v1 `silhouette.py` in as `cluster/resolution.py`. Populate `fast_auto_scrna/_native/__init__.py` to re-export the compiled submodules.
2. **V2-P3** — `uv venv`, `maturin develop --release`, `pytest tests/` 全绿.
3. **V2-P4** — complete docs (README module tour, INSTALL tested end-to-end, ROADMAP).
4. **GS-3** — implement Rust `silhouette_precomputed` kernel in `rust/crates/kernels/src/silhouette.rs`. Will cut the 222 k silhouette sweep from 890 s to ~20 s.
5. **Metric audit** — 222 k BBKNN silhouette curve was monotonic in k; investigate whether it's real signal (atlas-scale fine sub-clusters) or a graph-silhouette weakness (consider modularity / density-aware variants).

See [ROADMAP.md](ROADMAP.md) for the per-stage detail and performance baselines.

---

## Quick start

```bash
# 1) create env
cd F:/fast_auto_scRNA_v2
uv venv --python 3.10
source .venv/Scripts/activate      # Windows Git Bash
# or:  source .venv/bin/activate    # WSL / Linux

# 2) build Rust wheel + install editable
maturin develop --release -m rust/crates/py_bindings/Cargo.toml
pip install -e .

# 3) run tests
pytest tests/ -v
```

See [INSTALL.md](INSTALL.md) for the full setup (Windows + WSL) and
[ROADMAP.md](ROADMAP.md) for what's done / what's next.

## Layout — organized by pipeline stage

| # | Module | Responsibility |
|---|--------|----------------|
| 01 | `fast_auto_scrna/io/` | Load h5ad / rda / Seurat-qs; per-cell QC filtering |
| 02 | `fast_auto_scrna/preprocess/normalize.py` | normalize_total + log1p |
| 03 | `fast_auto_scrna/preprocess/hvg.py` | Highly-variable gene selection |
| 04 | `fast_auto_scrna/preprocess/scale.py` | z-score with max clip |
| 05 | `fast_auto_scrna/pca/` | Randomized PCA (Gavish-Donoho auto n_comps) — **Rust** |
| 06 | `fast_auto_scrna/integration/` | BBKNN / Harmony 2 / none — **Rust** |
| 07 | `fast_auto_scrna/neighbors/` | kNN + fuzzy_simplicial_set connectivities — **Rust** |
| 08 | `fast_auto_scrna/scib_metrics/` | iLISI / cLISI / graph_conn / kBET / silhouette — **Rust** |
| 09 | `fast_auto_scrna/umap/` | UMAP layout optimization — **Rust** |
| 10 | `fast_auto_scrna/cluster/` | Leiden + **graph-silhouette resolution selector** (new method) |
| 11 | `fast_auto_scrna/rogue/` | Per-cluster purity (entropy + loess) — **Rust** |
| — | `fast_auto_scrna/config.py` | `PipelineConfig` dataclass — every knob in one place |
| — | `fast_auto_scrna/runner.py` | `run_from_config(cfg, adata_in=None)` — main entry |
| — | `fast_auto_scrna/common/` | Shared sparse-matrix and I/O helpers |
| — | `fast_auto_scrna/_native/` | Thin re-export layer for compiled Rust bindings |

## Rust workspace

```
rust/
├── Cargo.toml                         workspace root
└── crates/
    ├── kernels/                       pure Rust algorithm kernels (rlib, no PyO3)
    │   └── src/
    │       ├── pca.rs
    │       ├── bbknn.rs
    │       ├── harmony/
    │       ├── umap.rs
    │       ├── fuzzy.rs               (fuzzy_simplicial_set)
    │       ├── metrics/               (lisi, graph_conn, kbet)
    │       ├── rogue.rs               (entropy_table + calculate_rogue)
    │       └── silhouette.rs          (graph silhouette — new, supersedes recall)
    └── py_bindings/                   PyO3 → fast_auto_scrna._native
```

## What's explicitly NOT in v2

- **`recall`** — scvalidate's recall cluster-number selector is dropped entirely.
  Cluster-number selection is now done by the graph-silhouette optimizer in
  `cluster/resolution.py`, which evaluates the same connectivity graph Leiden
  operates on and scales to the whole atlas.
- **`wilcoxon` / `knockoff` Rust kernels** — only used by recall, also dropped.
- **`RecallComparisonReport`** — no baseline-vs-recall report; the silhouette
  curve itself is the diagnostic.

## Testing data

- `data/pancreas_sub.rda` — 1000-cell pancreas lineage, 1 batch
  (symlink → `F:/NMF_rewrite/pancreas_sub.rda`). Unit-test canonical.
- `data/StepF.All_Cells.h5ad` — 222 529 cells × 20 055 genes prostate atlas,
  10 batches, ct.main (3-class) / ct.sub (7-class) / ct.sub.epi (13-class)
  ground-truth labels (symlink → `F:/NMF_rewrite/StepF.All_Cells.h5ad`).
  Atlas-scale canonical.

## History

v2 branched from v1 commit `c1107e8`. v1 lives at
`F:/NMF_rewrite/fast_auto_scRNA_v1/` (branch `v1`) and is now **deprecated** —
all new work happens here. v1 keeps the full recall / scvalidate history for
reference and is not deleted.

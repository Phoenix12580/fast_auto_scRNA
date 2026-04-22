# fast_auto_scRNA v2

End-to-end single-cell RNA-seq atlas pipeline — Rust-accelerated kernels with a
Python orchestration layer, organized **by pipeline stage** (not by legacy
library). Each stage is a self-contained module.

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

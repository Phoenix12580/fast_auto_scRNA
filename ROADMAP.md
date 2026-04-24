# ROADMAP — fast_auto_scRNA v2

## Status (2026-04-24, branch `main`)

**v2 is carved out of v1 into a single-root, stage-organized workspace.**
Current HEAD: `91e2aef` (V2-P1 kernels + bindings). Base commit: `c1107e8`
(v1 tip with GS-2 wiring + validated 222 k atlas smoke).

**V2-P0 done** (`b85c30b` + `32787b2` + branch rename):
- Worktree `F:/fast_auto_scRNA_v2` on branch `main` (renamed from `v2`)
- Skeleton: `fast_auto_scrna/` 11 stage subpackages + `rust/crates/{kernels,py_bindings}` workspace + empty `tests/benchmarks/docs/data`
- Docs: README / INSTALL / ROADMAP rewritten for v2
- Legacy: `scatlas/`, `scatlas_pipeline/`, `scvalidate_rewrite/`, `scripts/`, `docs/superpowers/`, `docs/images/`, `setup.sh`, `UPDATE.md` — 166 files, 26 426 lines deleted in single commit
- Old `main` branch preserved as tag `legacy-main-2026-04-22`
- Test data symlinked: `data/pancreas_sub.rda` + `data/StepF.All_Cells.h5ad`

**V2-P1 done** (`91e2aef`):
- `rust/crates/kernels/src/`: 7 modules migrated from v1 `scatlas-core`
  (`bbknn`, `fuzzy`, `harmony/`, `metrics/`, `pca`, `rogue`, `umap`);
  `rogue` promoted out of v1 `stats/` to a top-level module
- `rust/crates/py_bindings/src/`: PyO3 adapters from v1 `scatlas-py`
  re-registered under v2 stage names — `_native.{pca, bbknn, harmony,
  fuzzy, metrics, umap, rogue}`. BBKNN binding promoted out of v1
  `ext/` grab bag into its own submodule
- Dropped recall-only: `stats/wilcoxon.rs` (225 LOC) + `stats/knockoff.rs`
  (74 LOC) + their PyO3 wrappers
- `kernels/Cargo.toml`: added `libm = "0.2"` (used by `rogue::log1p`)
- `cargo check --workspace` + `cargo clippy --workspace --all-targets`: green

## Stages

| # | Stage | Code | Rust? | Status |
|---|-------|------|-------|--------|
| 01 | `io/` | load + qc + convert | — | ported-from-v1 |
| 02 | `preprocess/normalize` | normalize + log1p | no (scanpy) | ported |
| 03 | `preprocess/hvg` | HVG seurat/seurat_v3 | no (scanpy) | ported |
| 04 | `preprocess/scale` | z-score + max clip | no (scanpy) | ported |
| 05 | `pca/` | random SVD + Gavish-Donoho auto | ✅ | ported |
| 06 | `integration/bbknn` | HNSW + fuzzy | ✅ | ported |
| 06 | `integration/harmony` | Harmony 2 (6× R) | ✅ | ported |
| 06 | `integration/none` | plain kNN | — | ported |
| 07 | `neighbors/` | kNN + fuzzy_simplicial_set | ✅ | ported |
| 08 | `scib_metrics/` | iLISI/cLISI/graph_conn/kBET/silhouette | ✅ | ported |
| 09 | `umap/` | UMAP SGD layout | ✅ | ported |
| 10 | `cluster/leiden` | scanpy Leiden wrapper | — | ported |
| 10 | `cluster/resolution` | **graph-silhouette selector (NEW method)** | ⚠️ sklearn fallback | **port + Rust-ify next** |
| 11 | `rogue/` | entropy_table + calculate_rogue | ✅ | ported |

## Immediate TODO

1. **V2-P2**: carve `pipeline.py` (~1140 LOC) into stage modules under `fast_auto_scrna/`; populate `_native/__init__.py` to re-export the compiled submodules.
2. **V2-P3**: `.venv` + `maturin develop --release` + `pytest tests/` 全绿.
3. **V2-P4**: docs done (this file + README + INSTALL).
4. **V2-P5**: update memory; mark v1/scatlas/scvalidate_rewrite as DEPRECATED (do not delete).

## Next substantive work after reorganization

1. **GS-3**: Rust `silhouette_precomputed` kernel — the smoke test's 889.9 s sklearn
   fallback becomes ~20 s. Lands in `rust/crates/kernels/src/silhouette.rs`.
2. **GS-4**: wire the Python optimizer in `cluster/resolution.py` to dispatch to
   the Rust kernel when available.
3. **222k smoke re-run** with Rust kernel — expect total wall ~4 min (vs current 18 min).
4. **Harmony2 silhouette smoke** — compare Harmony-graph vs BBKNN-graph silhouette curves on 222k.
5. **Graph-silhouette metric audit** — 222k BBKNN smoke produced a monotonic curve; investigate whether that's true signal (more clusters = better graph fit) or a metric weakness. Consider modularity or density-aware variants.
6. **OOM-1**: wire `anndataoom` chunked preprocess (normalize / log1p / HVG / scale) — currently scanpy in-memory, limits atlas size.

## Explicitly dropped (from v1)

| Dropped | Reason |
|---------|--------|
| `scvalidate_rewrite/scvalidate/recall_py/*` | recall replaced by graph-silhouette |
| `wilcoxon` Rust kernel | only used by recall |
| `knockoff` Rust kernel | only used by recall |
| `recall_oom_backend` | recall-only |
| `RecallComparisonReport` | recall-only |
| `run_recall` config field + callers | recall-only |
| `test_recall_*` tests | recall-only |

## Performance baselines (222k StepF atlas, BBKNN route)

| Stage | Wall | Notes |
|-------|------|-------|
| load + qc | 19.5 s | anndata eager load |
| lognorm | 12.2 s | scanpy |
| hvg (seurat) | 24.6 s | scanpy |
| scale | 10.9 s | scanpy |
| **pca** (auto, Rust) | **82.8 s** | GD → 13 comps |
| **bbknn + fuzzy** (Rust + HNSW) | **35.1 s** | kNN 6.7M edges → fuzzy CSR 12.3M nnz |
| metrics | 1.4 s | scIB (no ground-truth labels in smoke) |
| **pipeline subtotal** | **187.5 s (3.1 min)** | |
| silhouette optimizer × 16 res × 50 iter | 889.9 s | **sklearn fallback — Rust-ify = GS-3** |
| **total smoke wall** | ~18 min | |

Graph-silhouette curve at 222k was monotonic; best in k ∈ [3, 10] was res=0.20 / k=10, silhouette = 0.00033 ± 0.00009. Needs metric audit.

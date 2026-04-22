# Recall Mandatory + anndata-oom OOM Solution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make recall the mandatory auto-cluster-k step in `fast_auto_scRNA_v1` pipeline, solve 157k OOM via anndata-oom backed storage, and emit a `RecallComparisonReport` so every run shows baseline-Leiden k vs recall-calibrated k.

**Architecture:** `find_clusters_recall` accepts an `AnnDataOOM` (or dense fallback for <30k). knockoff-augmented counts are written to a temp h5ad and read back as a backed AnnData, preprocess/PCA/neighbors are run once lazily via `anndataoom`, the while-loop only re-runs `sc.tl.leiden` and gene-chunked Wilcoxon via Rust kernel. Pipeline `_run_recall_for_route` always executes, produces per-route `RecallComparisonReport` into `adata.uns[f"recall_{method}_comparison"]`.

**Tech Stack:** Python 3.10+, `scvalidate_rust` (PyO3 0.24 Wilcoxon kernel, unchanged), `anndataoom` (lazy AnnData backing), `scanpy`/`anndata`, `pytest`. Dev env = WSL2.

**Spec reference:** `fast_auto_scRNA_v1/docs/superpowers/specs/2026-04-22-recall-mandatory-oom-design.md`

---

## File Structure

**v1 workspace** = new git worktree at `F:/NMF_rewrite/fast_auto_scRNA_v1/` on branch `v1` of `github.com/Phoenix12580/fast_auto_scRNA`.

**Files created** (all relative to `fast_auto_scRNA_v1/`):

- `scvalidate_rewrite/scvalidate/recall_py/_oom_backend.py` — new module holding oom-specific helpers (`_write_augmented_h5ad`, `_wilcoxon_pair_chunked`), so `core.py` stays readable
- `scvalidate_rewrite/scvalidate/recall_py/comparison.py` — new module with `RecallComparisonReport` dataclass + fate classifier
- `scvalidate_rewrite/tests/test_recall_oom_parity.py` — new test: 10k epithelia dense-vs-oom ARI ≥ 0.95
- `scvalidate_rewrite/tests/test_recall_oom_memory.py` — new test: 50k peak RSS < 8 GB
- `scvalidate_rewrite/tests/test_recall_comparison_report.py` — new test: end-to-end pancreas_sub produces valid report
- `scatlas_pipeline/benchmarks/recall_oom_157k.py` — new benchmark script

**Files modified**:

- `scvalidate_rewrite/scvalidate/recall_py/core.py` — `find_clusters_recall` accepts backed AnnData, preprocess moved out of while loop, `RecallResult` extended
- `scvalidate_rewrite/scvalidate/recall_py/__init__.py` — export `RecallComparisonReport`
- `scatlas_pipeline/pipeline.py` — delete `run_recall` field (line 142), always run recall (line 557-560), extend `_run_recall_for_route` with oom routing + comparison report
- `scatlas_pipeline/README.md` — update `run_recall` documentation
- `scatlas_pipeline/benchmarks/e2e_pancreas_sub.py`, `compare_panc8_full.py`, `compare_pancreas_sub.py`, `run_157k.py` — drop `run_recall=` arguments
- `scatlas_pipeline/configs/epithelia_157k.yaml` — drop `run_recall` key
- `scvalidate_rewrite/pyproject.toml` — add `anndataoom` as optional dependency
- `README.md` (top-level v1) — update `run_recall` descriptions + new comparison output section

---

## Task 1: Bootstrap v1 Worktree + Relocate Spec

**Files:**
- Setup: `F:/NMF_rewrite/fast_auto_scRNA_v1/` (worktree target)
- Move: spec + plan from temp `fast_auto_scRNA_v1/` pre-worktree files

**Context:** `F:/NMF_rewrite/fast_auto_scRNA/` is clean on `main`, tracking `https://github.com/Phoenix12580/fast_auto_scRNA`. The target `fast_auto_scRNA_v1/` directory already exists with spec+plan in `docs/superpowers/`, but it's NOT a git worktree yet. We need to temporarily stash spec/plan, create worktree, restore, and commit.

- [ ] **Step 1: Stash the pre-worktree spec + plan**

```bash
mv F:/NMF_rewrite/fast_auto_scRNA_v1 F:/NMF_rewrite/.v1_docs_stash
```

- [ ] **Step 2: Create worktree on new v1 branch**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA
git worktree add ../fast_auto_scRNA_v1 -b v1
```

Expected: "Preparing worktree (new branch 'v1')" + "HEAD is now at a4558bb feat: add per-cluster ROGUE bar plot"

- [ ] **Step 3: Restore spec + plan into worktree**

```bash
mkdir -p F:/NMF_rewrite/fast_auto_scRNA_v1/docs/superpowers/specs
mkdir -p F:/NMF_rewrite/fast_auto_scRNA_v1/docs/superpowers/plans
cp F:/NMF_rewrite/.v1_docs_stash/docs/superpowers/specs/*.md F:/NMF_rewrite/fast_auto_scRNA_v1/docs/superpowers/specs/
cp F:/NMF_rewrite/.v1_docs_stash/docs/superpowers/plans/*.md F:/NMF_rewrite/fast_auto_scRNA_v1/docs/superpowers/plans/
rm -rf F:/NMF_rewrite/.v1_docs_stash
```

- [ ] **Step 4: Verify worktree state**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1 && git status
```

Expected: "On branch v1" + "Untracked files: docs/superpowers/..."

- [ ] **Step 5: Commit spec + plan**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add docs/superpowers/
git commit -m "docs: v1 design spec + implementation plan for mandatory recall + OOM fix"
```

---

## Task 2: Verify anndataoom in WSL + pin dependency

**Files:**
- Modify: `fast_auto_scRNA_v1/scvalidate_rewrite/pyproject.toml`

**Context:** Memory (`reference_anndata_oom.md`) says anndataoom has Linux wheel; WSL is dev env per `feedback_wsl_primary.md`.

- [ ] **Step 1: Install anndataoom in active WSL venv**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1 && uv pip install anndataoom"
```

Expected: wheel installs cleanly, no compilation.

- [ ] **Step 2: Smoke-test import + chunked API**

```bash
wsl bash -c "python -c 'import anndataoom as oom; print(oom.__version__); import anndata; ad = anndata.AnnData(X=__import__(\"numpy\").random.rand(100, 50)); ad.write(\"/tmp/smoke.h5ad\"); b = oom.read(\"/tmp/smoke.h5ad\"); print(b.shape); import os; os.remove(\"/tmp/smoke.h5ad\")'"
```

Expected: version string + `(100, 50)` printed, no errors.

- [ ] **Step 3: Add to pyproject.toml as optional extra**

In `scvalidate_rewrite/pyproject.toml`, locate `[project.optional-dependencies]` and add:

```toml
[project.optional-dependencies]
oom = ["anndataoom>=0.1"]
```

If `[project.optional-dependencies]` doesn't exist, create the section just before `[tool.*]` blocks.

- [ ] **Step 4: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scvalidate_rewrite/pyproject.toml
git commit -m "deps: add anndataoom as optional extra for recall OOM backend"
```

---

## Task 3: Write Dense-vs-OOM Parity Test (Red)

**Files:**
- Create: `fast_auto_scRNA_v1/scvalidate_rewrite/tests/test_recall_oom_parity.py`

**Context:** Before touching core.py we lock in the parity contract: same seed + same data, dense and oom paths must give ARI ≥ 0.95 on 10k epithelia subset (the canonical benchmark).

- [ ] **Step 1: Write the failing test**

Create `scvalidate_rewrite/tests/test_recall_oom_parity.py`:

```python
"""Dense vs anndata-oom path parity for find_clusters_recall.

ARI ≥ 0.95 on a 10k synthetic dataset shaped like epithelia. Exact match
(bit-identical) is not required because chunked randomized PCA in the oom
path has small non-determinism even with fixed seeds.
"""
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score


@pytest.fixture
def counts_10k():
    rng = np.random.default_rng(0)
    # 2000 genes x 10000 cells, 5 latent groups
    n_genes, n_cells, n_groups = 2000, 10000, 5
    group = rng.integers(0, n_groups, n_cells)
    means = rng.gamma(2.0, 0.5, size=(n_groups, n_genes)).astype(np.float32)
    counts = rng.poisson(means[group].T).astype(np.int32)  # genes x cells
    return counts


def test_dense_vs_oom_parity(tmp_path, counts_10k):
    from scvalidate.recall_py import find_clusters_recall

    res_dense = find_clusters_recall(
        counts_10k,
        resolution_start=0.8, max_iterations=6,
        fdr=0.05, seed=0, verbose=False,
        backend="dense",
    )
    res_oom = find_clusters_recall(
        counts_10k,
        resolution_start=0.8, max_iterations=6,
        fdr=0.05, seed=0, verbose=False,
        backend="oom", scratch_dir=tmp_path,
    )
    ari = adjusted_rand_score(res_dense.labels, res_oom.labels)
    assert ari >= 0.95, f"dense-vs-oom ARI {ari:.3f} below 0.95"
    assert abs(len(np.unique(res_dense.labels)) - len(np.unique(res_oom.labels))) <= 1
```

- [ ] **Step 2: Run and confirm it fails with the right error**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_recall_oom_parity.py -v"
```

Expected: FAIL with `TypeError: find_clusters_recall() got an unexpected keyword argument 'backend'` (or similar — the `backend` arg doesn't exist yet).

- [ ] **Step 3: Commit the red test**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scvalidate_rewrite/tests/test_recall_oom_parity.py
git commit -m "test: add dense-vs-oom parity test for find_clusters_recall (red)"
```

---

## Task 4: Add `backend` Dispatch + Preserve Dense Path

**Files:**
- Modify: `fast_auto_scRNA_v1/scvalidate_rewrite/scvalidate/recall_py/core.py`

**Context:** Add a `backend: Literal["auto", "dense", "oom"] = "auto"` parameter. Route to dense (current code path) or oom (new, to be filled in later tasks). Auto picks based on n_cells threshold = 30_000.

- [ ] **Step 1: Read the current signature + docstring**

Inspect `core.py:203-257` to confirm current parameters before changing.

- [ ] **Step 2: Add backend parameter**

In `scvalidate_rewrite/scvalidate/recall_py/core.py:203`, change the signature and dispatch:

```python
def find_clusters_recall(
    counts_gxc,
    resolution_start: float = 0.8,
    reduction_percentage: float = 0.2,
    dims: int = 10,
    algorithm: str = "leiden",
    null_method: str = "ZIP",
    n_variable_features: int = 2000,
    fdr: float = 0.05,
    max_iterations: int = 20,
    seed: int | None = 0,
    verbose: bool = True,
    backend: str = "auto",              # new
    scratch_dir=None,                   # new, Path | None
    oom_threshold_cells: int = 30_000,  # new
) -> RecallResult:
    """..."""
    # -- backend resolution --
    if backend == "auto":
        n_cells = counts_gxc.shape[1] if hasattr(counts_gxc, "shape") else None
        backend = "oom" if (n_cells is not None and n_cells >= oom_threshold_cells) else "dense"
    if backend not in ("dense", "oom"):
        raise ValueError(f"backend must be 'auto'/'dense'/'oom', got {backend!r}")

    if backend == "oom":
        from scvalidate.recall_py._oom_backend import find_clusters_recall_oom
        return find_clusters_recall_oom(
            counts_gxc,
            resolution_start=resolution_start,
            reduction_percentage=reduction_percentage,
            dims=dims,
            algorithm=algorithm,
            null_method=null_method,
            n_variable_features=n_variable_features,
            fdr=fdr,
            max_iterations=max_iterations,
            seed=seed,
            verbose=verbose,
            scratch_dir=scratch_dir,
        )

    # -- dense path (existing code below, unchanged) --
    if sp.issparse(counts_gxc):
        counts = np.asarray(counts_gxc.todense())
    ...
```

Keep the rest of the function body identical (dense path).

- [ ] **Step 3: Create stub `_oom_backend.py` that raises NotImplementedError**

Create `scvalidate_rewrite/scvalidate/recall_py/_oom_backend.py`:

```python
"""anndata-oom backed path for find_clusters_recall."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.sparse as sp


def find_clusters_recall_oom(counts_gxc, **kwargs):
    raise NotImplementedError("oom backend — filled in Task 5")
```

- [ ] **Step 4: Run test — confirm it now fails deeper (NotImplementedError, not TypeError)**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_recall_oom_parity.py::test_dense_vs_oom_parity -v"
```

Expected: `NotImplementedError: oom backend — filled in Task 5`

- [ ] **Step 5: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scvalidate_rewrite/scvalidate/recall_py/core.py scvalidate_rewrite/scvalidate/recall_py/_oom_backend.py
git commit -m "refactor(recall): add backend=auto|dense|oom dispatch with stub oom path"
```

---

## Task 5: Implement `_write_augmented_h5ad` (Chunked Disk Write)

**Files:**
- Modify: `fast_auto_scRNA_v1/scvalidate_rewrite/scvalidate/recall_py/_oom_backend.py`
- Create: `fast_auto_scRNA_v1/scvalidate_rewrite/tests/test_oom_write_augmented.py`

**Context:** The knockoff-augmented counts matrix (2G × N int32) at 157k is 20.5 GB. Write to h5ad in cell-wise chunks so peak RAM ≈ 1 chunk = 2G × chunk_cells × 4B.

- [ ] **Step 1: Write the failing unit test**

Create `scvalidate_rewrite/tests/test_oom_write_augmented.py`:

```python
"""Unit test for _write_augmented_h5ad."""
import numpy as np
import anndata
from pathlib import Path


def test_write_augmented_roundtrip(tmp_path: Path):
    from scvalidate.recall_py._oom_backend import _write_augmented_h5ad

    rng = np.random.default_rng(0)
    G, N = 500, 2000
    real = rng.poisson(1.0, size=(G, N)).astype(np.int32)
    knock = rng.poisson(1.0, size=(G, N)).astype(np.int32)

    out = tmp_path / "aug.h5ad"
    _write_augmented_h5ad(real, knock, out, chunk_cells=500)

    ad = anndata.read_h5ad(out)
    # Stored as cells x (2G) to match anndata convention
    assert ad.shape == (N, 2 * G)
    # First G columns are real (transposed), last G are knockoff
    np.testing.assert_array_equal(ad.X[:, :G].T.astype(np.int32), real)
    np.testing.assert_array_equal(ad.X[:, G:].T.astype(np.int32), knock)
    assert "is_knockoff" in ad.var.columns
    assert ad.var["is_knockoff"].iloc[:G].eq(False).all()
    assert ad.var["is_knockoff"].iloc[G:].eq(True).all()
```

- [ ] **Step 2: Run — confirm it fails (function missing)**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_oom_write_augmented.py -v"
```

Expected: `ImportError: cannot import name '_write_augmented_h5ad'`

- [ ] **Step 3: Implement `_write_augmented_h5ad`**

In `scvalidate_rewrite/scvalidate/recall_py/_oom_backend.py`, add:

```python
import anndata
import pandas as pd


def _write_augmented_h5ad(
    real_gxc: np.ndarray,
    knock_gxc: np.ndarray,
    out_path: Path,
    chunk_cells: int = 5000,
) -> None:
    """Write [real | knockoff] stacked gene-wise to disk as AnnData cells × (2G).

    Streams N cells in chunks so RAM peak ≈ 2G × chunk_cells × 4B.
    Stored transposed to match scanpy's cells × genes convention.
    """
    assert real_gxc.shape == knock_gxc.shape
    G, N = real_gxc.shape
    # Build var table labelling real/knockoff rows
    var = pd.DataFrame({
        "is_knockoff": [False] * G + [True] * G,
    }, index=[f"g{i}" for i in range(G)] + [f"k{i}" for i in range(G)])
    obs = pd.DataFrame(index=[f"c{i}" for i in range(N)])

    # First write header with empty X of correct shape, then fill column-wise
    # anndata's write_h5ad supports passing a pre-built AnnData with chunked X
    # via h5py. Simplest reliable method: build small dense slices and concat.
    import h5py
    with h5py.File(out_path, "w") as f:
        # /X as cells x (2G) float32 chunked
        dset = f.create_dataset(
            "X", shape=(N, 2 * G), dtype=np.float32,
            chunks=(min(chunk_cells, N), 2 * G),
            compression=None,
        )
        for start in range(0, N, chunk_cells):
            end = min(start + chunk_cells, N)
            # real_gxc[:, start:end].T  -> (end-start, G)
            chunk = np.empty((end - start, 2 * G), dtype=np.float32)
            chunk[:, :G] = real_gxc[:, start:end].T.astype(np.float32, copy=False)
            chunk[:, G:] = knock_gxc[:, start:end].T.astype(np.float32, copy=False)
            dset[start:end, :] = chunk
        # Write minimal AnnData h5 structure so anndata.read_h5ad works
        f.create_group("obs"); f.create_group("var"); f.create_group("uns")
        f.create_group("obsm"); f.create_group("varm"); f.create_group("obsp")
        f.create_group("varp"); f.create_group("layers")
    # Re-open via anndata to attach obs/var properly (AnnData format v0.8+)
    ad = anndata.read_h5ad(out_path)
    ad.obs = obs
    ad.var = var
    ad.write_h5ad(out_path)
```

- [ ] **Step 4: Run the test — confirm pass**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_oom_write_augmented.py -v"
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scvalidate_rewrite/scvalidate/recall_py/_oom_backend.py scvalidate_rewrite/tests/test_oom_write_augmented.py
git commit -m "feat(recall/oom): implement _write_augmented_h5ad with cell-wise chunking"
```

---

## Task 6: Implement `_wilcoxon_pair_chunked` (Gene-Axis Streaming)

**Files:**
- Modify: `fast_auto_scRNA_v1/scvalidate_rewrite/scvalidate/recall_py/_oom_backend.py`
- Create: `fast_auto_scRNA_v1/scvalidate_rewrite/tests/test_wilcoxon_pair_chunked.py`

**Context:** Wilcoxon is per-gene independent — chunking over genes lets us keep only a slab in memory. Rust kernel `scvalidate_rust.wilcoxon_ranksum_matrix` is unchanged.

- [ ] **Step 1: Write failing test — parity with dense `_wilcoxon_per_gene`**

Create `scvalidate_rewrite/tests/test_wilcoxon_pair_chunked.py`:

```python
"""_wilcoxon_pair_chunked must match _wilcoxon_per_gene bit-close on dense input."""
import numpy as np


def test_chunked_matches_dense():
    from scvalidate.recall_py._oom_backend import _wilcoxon_pair_chunked_dense
    from scvalidate.recall_py.core import _wilcoxon_per_gene

    rng = np.random.default_rng(42)
    G, N = 1000, 500
    log_counts = rng.poisson(1.0, size=(G, N)).astype(np.float32)
    log_counts = np.log1p(log_counts)
    mask1 = rng.random(N) < 0.4
    mask2 = ~mask1

    p_full = _wilcoxon_per_gene(log_counts, mask1, mask2)
    p_chunked = _wilcoxon_pair_chunked_dense(log_counts, mask1, mask2, chunk=250)

    np.testing.assert_allclose(p_full, p_chunked, atol=1e-10)
```

- [ ] **Step 2: Run — confirm fail**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_wilcoxon_pair_chunked.py -v"
```

Expected: ImportError.

- [ ] **Step 3: Implement dense helper (used by test) + oom version**

Add to `_oom_backend.py`:

```python
def _wilcoxon_pair_chunked_dense(
    log_counts_gxc: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    chunk: int = 2000,
) -> np.ndarray:
    """Chunked per-gene Wilcoxon on a dense ndarray, for parity testing.

    Real production path uses _wilcoxon_pair_chunked_oom (reads h5ad chunks).
    Both call the same Rust kernel.
    """
    from scvalidate.recall_py.core import _wilcoxon_per_gene
    G = log_counts_gxc.shape[0]
    out = np.empty(G, dtype=np.float64)
    for start in range(0, G, chunk):
        end = min(start + chunk, G)
        out[start:end] = _wilcoxon_per_gene(
            log_counts_gxc[start:end], mask1, mask2,
        )
    return out


def _wilcoxon_pair_chunked_oom(
    log_counts_adata,         # backed AnnData (cells x 2G)
    mask1: np.ndarray,        # cell-level boolean
    mask2: np.ndarray,
    chunk_genes: int = 2000,
) -> np.ndarray:
    """Stream gene chunks from a backed AnnData, run Rust Wilcoxon, concat p."""
    from scvalidate.recall_py.core import _wilcoxon_per_gene
    n_vars = log_counts_adata.n_vars  # == 2G
    out = np.empty(n_vars, dtype=np.float64)
    for start in range(0, n_vars, chunk_genes):
        end = min(start + chunk_genes, n_vars)
        # Pull dense slab genes[start:end] for all cells (cells x chunk)
        # Convert to genes x cells for _wilcoxon_per_gene convention
        sl = log_counts_adata[:, start:end].X
        if hasattr(sl, "toarray"):
            sl = sl.toarray()
        sl_gxc = np.asarray(sl).T
        out[start:end] = _wilcoxon_per_gene(sl_gxc, mask1, mask2)
    return out
```

- [ ] **Step 4: Run test — confirm pass**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_wilcoxon_pair_chunked.py -v"
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scvalidate_rewrite/scvalidate/recall_py/_oom_backend.py scvalidate_rewrite/tests/test_wilcoxon_pair_chunked.py
git commit -m "feat(recall/oom): implement chunked Wilcoxon (dense + backed variants)"
```

---

## Task 7: Implement `find_clusters_recall_oom` Orchestrator

**Files:**
- Modify: `fast_auto_scRNA_v1/scvalidate_rewrite/scvalidate/recall_py/_oom_backend.py`

**Context:** Replace the stub with the full oom path: generate knockoffs → write augmented → oom-backed preprocess (ONCE, outside the while loop) → PCA + neighbors (ONCE) → while-loop does only leiden + chunked Wilcoxon.

- [ ] **Step 1: Implement `find_clusters_recall_oom`**

In `_oom_backend.py`, replace the stub with:

```python
import tempfile
from scvalidate.recall_py.knockoff import (
    generate_knockoff_matrix, knockoff_threshold_offset1,
)
from scvalidate.recall_py.core import RecallResult


def find_clusters_recall_oom(
    counts_gxc,
    *,
    resolution_start: float = 0.8,
    reduction_percentage: float = 0.2,
    dims: int = 10,
    algorithm: str = "leiden",
    null_method: str = "ZIP",
    n_variable_features: int = 2000,
    fdr: float = 0.05,
    max_iterations: int = 20,
    seed: int | None = 0,
    verbose: bool = True,
    scratch_dir=None,
) -> RecallResult:
    """anndata-oom backed find_clusters_recall.

    Keeps the 2G × N augmented matrix on disk (scratch h5ad). Preprocess,
    PCA, neighbors computed once outside the while loop. Inside the loop
    only sc.tl.leiden re-runs. Wilcoxon streams gene chunks from disk.
    """
    import anndataoom as oom
    import scanpy as sc

    if sp.issparse(counts_gxc):
        counts = np.asarray(counts_gxc.todense())
    else:
        counts = np.asarray(counts_gxc)
    counts = counts.astype(np.int32)
    G, N = counts.shape

    # -- scratch --
    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmpdir:
        tmpdir = Path(tmpdir)
        aug_path = tmpdir / "augmented.h5ad"

        if verbose:
            print(f"[recall/oom] generating knockoffs (G={G}, N={N}) ...")
        knock = generate_knockoff_matrix(
            counts, null_method=null_method, seed=seed, verbose=verbose,
        ).astype(np.int32)
        _write_augmented_h5ad(counts, knock, aug_path, chunk_cells=5000)
        del knock, counts  # reclaim

        if verbose:
            print(f"[recall/oom] oom.read + lazy preprocess ...")
        aug = oom.read(str(aug_path))  # backed, cells x 2G

        # Lazy normalize + log1p
        aug_log = oom.log1p(oom.normalize_total(aug, target_sum=1e4))
        # HVG on augmented — allow 2 * n_variable_features because half are knockoffs
        n_hvg = min(2 * n_variable_features, aug_log.n_vars)
        sc.pp.highly_variable_genes(aug_log, n_top_genes=n_hvg, flavor="seurat")
        aug_hvg = aug_log[:, aug_log.var["highly_variable"]]

        # PCA (chunked randomized SVD via oom) + neighbors — ONCE
        n_pcs = min(dims, aug_hvg.n_vars - 1, aug_hvg.n_obs - 1)
        # anndataoom's chunked PCA materializes only X_pca (n_cells x n_pcs)
        sc.pp.scale(aug_hvg, max_value=10)     # oom chunked path in newer oom
        sc.tl.pca(aug_hvg, n_comps=n_pcs, random_state=seed)
        sc.pp.neighbors(aug_hvg, n_pcs=n_pcs, random_state=seed)

        # While loop: only leiden + per-pair chunked Wilcoxon
        resolution = float(resolution_start)
        n_iter = 0
        last_labels = None
        resolution_trajectory: list[float] = []
        k_trajectory: list[int] = []
        converged = False
        n_real = G

        while n_iter < max_iterations:
            n_iter += 1
            resolution_trajectory.append(resolution)
            if verbose:
                print(f"[recall/oom] iter {n_iter}: res={resolution:.4f}")

            if algorithm == "leiden":
                sc.tl.leiden(
                    aug_hvg, resolution=resolution, random_state=seed,
                    flavor="igraph", n_iterations=2, directed=False,
                )
                labels = aug_hvg.obs["leiden"].astype(int).to_numpy()
            elif algorithm == "louvain":
                sc.tl.louvain(aug_hvg, resolution=resolution, random_state=seed)
                labels = aug_hvg.obs["louvain"].astype(int).to_numpy()
            else:
                raise ValueError(f"algorithm {algorithm!r}")
            last_labels = labels
            k = int(labels.max()) + 1
            k_trajectory.append(k)

            if k < 2:
                if verbose: print(f"[recall/oom] single cluster — stop")
                converged = True
                break

            found_merged = False
            for i in range(k):
                for j in range(i):
                    m1 = labels == i
                    m2 = labels == j
                    pvals = _wilcoxon_pair_chunked_oom(aug_log, m1, m2)
                    p_real = pvals[:n_real]; p_knock = pvals[n_real:]
                    w = -np.log10(p_real) - (-np.log10(p_knock))
                    t = knockoff_threshold_offset1(w, fdr=fdr)
                    n_sel = int((w >= t).sum()) if np.isfinite(t) else 0
                    if n_sel == 0:
                        found_merged = True
                        if verbose: print(f"[recall/oom]   pair ({i},{j}) 0 sel")
                        break
                if found_merged:
                    break

            if not found_merged:
                converged = True
                if verbose: print(f"[recall/oom] converged at k={k}")
                break
            resolution = (1 - reduction_percentage) * resolution

        assert last_labels is not None
        per_cluster_pass = {int(c): True for c in np.unique(last_labels)}
        return RecallResult(
            labels=last_labels,
            resolution=resolution,
            n_iterations=n_iter,
            per_cluster_pass=per_cluster_pass,
            resolution_trajectory=resolution_trajectory,
            k_trajectory=k_trajectory,
            converged=converged,
        )
```

- [ ] **Step 2: Extend `RecallResult` dataclass in `core.py`**

In `scvalidate_rewrite/scvalidate/recall_py/core.py`, update the dataclass (lines 32-57):

```python
@dataclass
class RecallResult:
    labels: np.ndarray
    resolution: float
    n_iterations: int
    per_cluster_pass: dict[int, bool]
    # v1 additions:
    resolution_trajectory: list[float] = field(default_factory=list)
    k_trajectory: list[int] = field(default_factory=list)
    converged: bool = False
```

Add `from dataclasses import field` to imports if not already there.

Also update the dense path at the end of `find_clusters_recall` to populate the new fields (so dense and oom have the same result shape):

In the dense while loop (around line 293), add a `resolution_trajectory: list[float] = []` and `k_trajectory: list[int] = []` before the loop; append inside; set `converged = True` on the two break paths where convergence happened (not on `max_iterations` exhaust). Pass to RecallResult.

- [ ] **Step 3: Run the parity test from Task 3**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_recall_oom_parity.py -v"
```

Expected: PASS with ARI ≥ 0.95.

- [ ] **Step 4: If it fails with < 0.95 ARI**

Debug by adding `verbose=True`, compare `k_trajectory` between dense and oom. Most likely cause: HVG selection differs because the dense path's scanpy wrapper picks HVG differently from oom's chunked variance. Acceptable fix: force same HVG set in both paths by pre-computing HVG once and feeding the same list to both.

- [ ] **Step 5: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scvalidate_rewrite/scvalidate/recall_py/
git commit -m "feat(recall): oom backend orchestrator + RecallResult trajectory fields"
```

---

## Task 8: `RecallComparisonReport` Dataclass + Fate Classifier

**Files:**
- Create: `fast_auto_scRNA_v1/scvalidate_rewrite/scvalidate/recall_py/comparison.py`
- Modify: `fast_auto_scRNA_v1/scvalidate_rewrite/scvalidate/recall_py/__init__.py`
- Create: `fast_auto_scRNA_v1/scvalidate_rewrite/tests/test_recall_comparison_unit.py`

**Context:** This is the "对比" deliverable — every pipeline run ships a report comparing baseline Leiden k (from `leiden_target_n` rule) vs recall-calibrated k.

- [ ] **Step 1: Write failing unit test**

Create `scvalidate_rewrite/tests/test_recall_comparison_unit.py`:

```python
import numpy as np


def test_comparison_report_basic():
    from scvalidate.recall_py.comparison import build_comparison_report

    rng = np.random.default_rng(0)
    # Baseline has 10 clusters, recall merged it to 6
    labels_baseline = rng.integers(0, 10, size=1000)
    # recall: first 5 clusters kept, last 5 all merged into cluster 5
    labels_recall = np.where(labels_baseline < 5, labels_baseline, 5)

    rep = build_comparison_report(
        labels_baseline=labels_baseline,
        labels_recall=labels_recall,
        resolution_baseline=0.8,
        resolution_recall=0.4,
        recall_converged=True,
        k_trajectory=[10, 8, 6],
        recall_wall_time_s=123.4,
    )
    assert rep.k_baseline == 10
    assert rep.k_recall == 6
    assert rep.delta_k == 4
    assert 0.3 <= rep.ari_baseline_vs_recall <= 0.8
    # clusters 0..4 kept, 5..9 merged
    fates = rep.per_baseline_cluster_fate
    assert all("kept" in fates[c] for c in range(5))
    assert all("merged" in fates[c] for c in range(5, 10))
```

- [ ] **Step 2: Run — fails on ImportError**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_recall_comparison_unit.py -v"
```

- [ ] **Step 3: Implement `comparison.py`**

```python
"""RecallComparisonReport — baseline Leiden k vs recall-calibrated k."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from collections import Counter
import numpy as np
from sklearn.metrics import adjusted_rand_score


@dataclass
class RecallComparisonReport:
    k_baseline: int
    k_recall: int
    resolution_baseline: float
    resolution_recall: float
    delta_k: int
    ari_baseline_vs_recall: float
    recall_converged: bool
    per_baseline_cluster_fate: dict            # {cluster_id: "kept"|"merged_with_{X}"|"split"}
    k_trajectory: list                          # recall per-iter k
    recall_wall_time_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_fate(labels_baseline: np.ndarray, labels_recall: np.ndarray) -> dict:
    """For each baseline cluster, describe what recall did to it.

    'kept'             — >= 80% of baseline cells stayed in one recall cluster
                         that nobody else dominates.
    'merged_with_{X}'  — >= 80% moved to a recall cluster that also absorbs
                         other baseline clusters.
    'split_into_[X,Y]' — baseline cluster is split across >=2 recall clusters
                         each holding >= 20%.
    """
    fates: dict = {}
    recall_dominance = {}  # recall_cluster -> set of baseline clusters it dominates
    for b in np.unique(labels_baseline):
        mask = labels_baseline == b
        counter = Counter(labels_recall[mask])
        total = mask.sum()
        top_recall, top_n = counter.most_common(1)[0]
        if top_n / total >= 0.80:
            recall_dominance.setdefault(int(top_recall), set()).add(int(b))
        else:
            # splits: recall clusters with >= 20% share
            splits = [r for r, n in counter.items() if n / total >= 0.20]
            fates[int(b)] = f"split_into_{sorted(splits)}"

    for r, bs in recall_dominance.items():
        if len(bs) == 1:
            b = next(iter(bs))
            fates[b] = "kept"
        else:
            for b in bs:
                others = sorted(bs - {b})
                fates[b] = f"merged_with_{others}"
    return fates


def build_comparison_report(
    *,
    labels_baseline: np.ndarray,
    labels_recall: np.ndarray,
    resolution_baseline: float,
    resolution_recall: float,
    recall_converged: bool,
    k_trajectory: list,
    recall_wall_time_s: float,
) -> RecallComparisonReport:
    k_b = int(len(np.unique(labels_baseline)))
    k_r = int(len(np.unique(labels_recall)))
    ari = float(adjusted_rand_score(labels_baseline, labels_recall))
    fates = _classify_fate(labels_baseline, labels_recall)
    return RecallComparisonReport(
        k_baseline=k_b,
        k_recall=k_r,
        resolution_baseline=resolution_baseline,
        resolution_recall=resolution_recall,
        delta_k=k_b - k_r,
        ari_baseline_vs_recall=ari,
        recall_converged=recall_converged,
        per_baseline_cluster_fate=fates,
        k_trajectory=list(k_trajectory),
        recall_wall_time_s=recall_wall_time_s,
    )
```

- [ ] **Step 4: Export from `__init__.py`**

In `scvalidate_rewrite/scvalidate/recall_py/__init__.py`, add:

```python
from scvalidate.recall_py.comparison import RecallComparisonReport, build_comparison_report

__all__ = [
    "RecallResult",
    ...,
    "RecallComparisonReport",
    "build_comparison_report",
]
```

Append the two new names to the existing `__all__` list.

- [ ] **Step 5: Run test — pass**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_recall_comparison_unit.py -v"
```

- [ ] **Step 6: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scvalidate_rewrite/scvalidate/recall_py/comparison.py scvalidate_rewrite/scvalidate/recall_py/__init__.py scvalidate_rewrite/tests/test_recall_comparison_unit.py
git commit -m "feat(recall): RecallComparisonReport + fate classifier"
```

---

## Task 9: Delete `run_recall` from PipelineConfig + Update Call Site

**Files:**
- Modify: `fast_auto_scRNA_v1/scatlas_pipeline/pipeline.py:142, 557-560`
- Modify: `fast_auto_scRNA_v1/scatlas_pipeline/README.md:44`
- Modify: `fast_auto_scRNA_v1/scatlas_pipeline/configs/epithelia_157k.yaml:62`
- Modify: all `fast_auto_scRNA_v1/scatlas_pipeline/benchmarks/*.py` with `run_recall=`

**Context:** Recall is mandatory in v1. The parameter is removed; callers passing it hit `TypeError`. Since dataclass `__init__` is auto-generated, removing the field automatically gives the right error.

- [ ] **Step 1: Write test that the field is gone**

Append to `scvalidate_rewrite/tests/test_recall_comparison_unit.py` (or create `test_pipeline_config.py` if you prefer):

```python
def test_pipeline_config_rejects_run_recall():
    import pytest
    from scatlas_pipeline.pipeline import PipelineConfig
    with pytest.raises(TypeError, match="run_recall"):
        PipelineConfig(run_recall=True)
```

- [ ] **Step 2: Run — confirm fail (since field still exists, test fails)**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_recall_comparison_unit.py::test_pipeline_config_rejects_run_recall -v"
```

Expected: FAIL (PipelineConfig accepts the argument).

- [ ] **Step 3: Delete the field**

In `scatlas_pipeline/pipeline.py:142`, remove:
```python
    run_recall: bool = False
```
and keep the next three `recall_*` config fields as-is.

Add after `recall_max_iterations`:
```python
    recall_scratch_dir: str | None = None  # None → tempfile default
```

- [ ] **Step 4: Remove the `if cfg.run_recall` gate**

In `scatlas_pipeline/pipeline.py:556-560`, replace:
```python
    # --- 11 recall (optional) ----------------------------------------------
    if cfg.run_recall:
        t0 = time.perf_counter()
        _run_recall_for_route(adata, method, cfg)
        route_t["recall"] = _step(f"11 {method}/recall", t0) - t0
```
with:
```python
    # --- 11 recall (mandatory in v1) ---------------------------------------
    t0 = time.perf_counter()
    _run_recall_for_route(adata, method, cfg)
    route_t["recall"] = _step(f"11 {method}/recall", t0) - t0
```

- [ ] **Step 5: Drop `run_recall=...` from all benchmarks + yaml**

Use an Edit tool for each:
- `scatlas_pipeline/benchmarks/run_157k.py:22` — delete the line `run_recall=False,`
- `scatlas_pipeline/benchmarks/e2e_pancreas_sub.py:83` — delete `run_recall=False,`
- `scatlas_pipeline/benchmarks/compare_panc8_full.py:95` — delete `run_recall=True,`
- `scatlas_pipeline/configs/epithelia_157k.yaml:62` — delete `run_recall: false`
- `scatlas_pipeline/README.md:44` — delete the `run_recall=False` doc line, update its surrounding text to note recall is now mandatory
- `scatlas_pipeline/benchmarks/compare_pancreas_sub.py` — leaves intact (it calls `find_clusters_recall` directly, doesn't go through PipelineConfig)

- [ ] **Step 6: Run the TypeError test + full unit tests**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/ -v"
```

Expected: new `test_pipeline_config_rejects_run_recall` passes; no other tests regressed.

- [ ] **Step 7: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scatlas_pipeline/ scvalidate_rewrite/tests/
git commit -m "feat(pipeline): delete run_recall flag — recall is mandatory in v1"
```

---

## Task 10: Wire `_run_recall_for_route` to Oom Backend + Build Comparison Report

**Files:**
- Modify: `fast_auto_scRNA_v1/scatlas_pipeline/pipeline.py:753-780`

**Context:** The pipeline helper currently calls `find_clusters_recall` with counts and writes labels back. We need to: (a) pass `backend="auto"` + `scratch_dir` from config, (b) grab baseline Leiden labels from `obs[f"leiden_{method}"]`, (c) build the comparison report, (d) stash into `adata.uns`.

- [ ] **Step 1: Rewrite `_run_recall_for_route`**

In `scatlas_pipeline/pipeline.py`, replace the body of `_run_recall_for_route` (line 753-780) with:

```python
def _run_recall_for_route(adata, method: str, cfg: PipelineConfig) -> None:
    """scvalidate recall on raw counts with oom backend for >=30k cells.

    Also emits RecallComparisonReport comparing:
      - baseline Leiden k (from step 10, selected by cfg.leiden_target_n)
      - recall-calibrated k (this step)
    stored in adata.uns[f"recall_{method}_comparison"].
    """
    import time
    try:
        from scvalidate.recall_py import (
            find_clusters_recall, build_comparison_report,
        )
    except ImportError:
        print(f"         [{method}] [recall] scvalidate not installed — skipping")
        return

    # Raw counts in layers["counts"] (preserved before lognorm)
    X = adata.layers["counts"]
    counts_gxc = (X if sp.issparse(X) else sp.csr_matrix(X)).T

    # Baseline labels from step 10 (Leiden + target_n)
    baseline_key = f"leiden_{method}"
    if baseline_key not in adata.obs.columns:
        print(f"         [{method}] [recall] no baseline leiden — skipping")
        return
    labels_baseline = adata.obs[baseline_key].astype(int).to_numpy()
    # Find the selected baseline resolution (stored by step 10)
    res_baseline = float(adata.uns.get(f"leiden_{method}_resolution", 0.8))

    t0 = time.perf_counter()
    result = find_clusters_recall(
        counts_gxc,
        resolution_start=cfg.recall_resolution_start,
        fdr=cfg.recall_fdr,
        max_iterations=cfg.recall_max_iterations,
        seed=0,
        backend="auto",
        scratch_dir=cfg.recall_scratch_dir,
    )
    wall = time.perf_counter() - t0

    adata.obs[f"recall_{method}"] = result.labels.astype(str)
    adata.uns[f"recall_{method}_resolution"] = result.resolution
    adata.uns[f"recall_{method}_iterations"] = result.n_iterations

    # Comparison report
    report = build_comparison_report(
        labels_baseline=labels_baseline,
        labels_recall=result.labels,
        resolution_baseline=res_baseline,
        resolution_recall=result.resolution,
        recall_converged=result.converged,
        k_trajectory=result.k_trajectory,
        recall_wall_time_s=wall,
    )
    adata.uns[f"recall_{method}_comparison"] = report.to_dict()

    print(
        f"         [{method}] recall: k_baseline={report.k_baseline} → "
        f"k_recall={report.k_recall} (ΔK={report.delta_k}), "
        f"ARI={report.ari_baseline_vs_recall:.3f}, "
        f"converged={report.recall_converged}, "
        f"wall={wall:.1f}s"
    )
```

- [ ] **Step 2: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scatlas_pipeline/pipeline.py
git commit -m "feat(pipeline): route recall through oom backend + emit comparison report"
```

---

## Task 11: End-to-End Test on pancreas_sub

**Files:**
- Create: `fast_auto_scRNA_v1/scvalidate_rewrite/tests/test_recall_comparison_report.py`

**Context:** Verify that running the full pipeline on the canonical 1k dataset produces a valid `recall_*_comparison` entry in `adata.uns`.

- [ ] **Step 1: Write the test**

```python
"""End-to-end: pancreas_sub pipeline emits recall comparison report."""
from pathlib import Path
import pytest


PANCREAS_SUB = Path("/mnt/f/NMF_rewrite/pancreas_sub.rda")


@pytest.mark.skipif(not PANCREAS_SUB.exists(), reason="canonical data missing")
def test_pipeline_produces_recall_comparison(tmp_path):
    from scatlas_pipeline import run_pipeline, PipelineConfig

    cfg = PipelineConfig(
        integration="none",
        label_key=None,
        run_leiden=True,
        leiden_resolutions=[0.5, 0.8, 1.0],
        leiden_target_n=(4, 12),
        recall_max_iterations=6,
        out_h5ad=str(tmp_path / "out.h5ad"),
    )
    adata = run_pipeline(str(PANCREAS_SUB), batch_key="_dummy", cfg=cfg)

    # At least one route should have produced a comparison report
    keys = [k for k in adata.uns.keys() if k.startswith("recall_") and k.endswith("_comparison")]
    assert keys, f"no recall_*_comparison in uns; keys: {list(adata.uns.keys())}"

    rep = adata.uns[keys[0]]
    assert rep["k_baseline"] >= 2
    assert rep["k_recall"] >= 1
    assert -5 <= rep["delta_k"] <= 20  # sanity
    assert 0.0 <= rep["ari_baseline_vs_recall"] <= 1.0
    assert isinstance(rep["k_trajectory"], list) and len(rep["k_trajectory"]) >= 1
    assert rep["recall_wall_time_s"] > 0.0
```

- [ ] **Step 2: Run the test**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_recall_comparison_report.py -v"
```

Expected: PASS (or skip if pancreas_sub.rda missing; file should be present per `project_canonical_test_data.md`).

- [ ] **Step 3: If it fails**

Most likely: `run_pipeline` API mismatch — check `scatlas_pipeline/__init__.py` to confirm the function name and signature. Adjust test to match.

- [ ] **Step 4: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scvalidate_rewrite/tests/test_recall_comparison_report.py
git commit -m "test: end-to-end pancreas_sub produces recall comparison report"
```

---

## Task 12: Memory Benchmark Test at 50k

**Files:**
- Create: `fast_auto_scRNA_v1/scvalidate_rewrite/tests/test_recall_oom_memory.py`

**Context:** Enforces the memory promise. dense path baseline at 50k was 34 GB (per memory); oom path must be < 8 GB to prove the win.

- [ ] **Step 1: Write the test**

```python
"""Memory smoke test: oom backend at 50k cells peak RSS < 8 GB.

Reference (from v0.4 progress memory):
    dense 50k f32 peak RSS = 34 GB
    oom target             = < 8 GB  (≥ 4× reduction)
"""
import os
import resource
import pytest
import numpy as np


@pytest.mark.slow
def test_oom_memory_50k(tmp_path):
    from scvalidate.recall_py import find_clusters_recall
    rng = np.random.default_rng(0)
    G, N = 8000, 50_000
    # Synthetic sparse-like counts
    counts = rng.poisson(0.3, size=(G, N)).astype(np.int32)

    # Reset peak counter (Linux only — resource.RUSAGE_SELF.ru_maxrss is kB)
    pre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    _ = find_clusters_recall(
        counts,
        resolution_start=0.8, max_iterations=3,
        fdr=0.05, seed=0, verbose=False,
        backend="oom", scratch_dir=tmp_path,
    )

    post = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = post / (1024 * 1024)  # kB → GB on Linux
    print(f"peak RSS delta: {(post - pre)/(1024*1024):.2f} GB; absolute: {peak_gb:.2f} GB")
    assert peak_gb < 8.0, f"oom backend 50k peak {peak_gb:.2f} GB exceeds 8 GB budget"
```

Note: on Windows `ru_maxrss` is in bytes; the test is Linux/WSL only — which is fine since WSL is our dev env.

- [ ] **Step 2: Run the slow test**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/test_recall_oom_memory.py -v -m slow -s"
```

Expected: PASS with peak RSS reported < 8 GB. Takes ~5-10 min.

- [ ] **Step 3: If it exceeds 8 GB**

Investigate: (a) confirm `counts` input itself isn't 12 GB (at G=8000, N=50k, int32 it's 1.6 GB — OK); (b) check `aug.h5ad` size vs in-memory copies; (c) ensure `del knock, counts` actually runs; (d) consider lowering `chunk_genes` in Wilcoxon.

- [ ] **Step 4: Commit**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scvalidate_rewrite/tests/test_recall_oom_memory.py
git commit -m "test: oom backend 50k peak RSS < 8 GB"
```

---

## Task 13: 157k Full-Data Benchmark Script

**Files:**
- Create: `fast_auto_scRNA_v1/scatlas_pipeline/benchmarks/recall_oom_157k.py`

**Context:** End-to-end ship validation. Data is `F:/NMF_rewrite/epithelia_full.h5ad` per workspace layout.

- [ ] **Step 1: Write the benchmark script**

```python
"""Recall OOM benchmark on 157k epithelia full dataset.

Success criteria:
  * recall runs to completion (no OOM)
  * peak RSS < 5 GB for recall step (excluding raw X)
  * recall wall < 50 min (including first I/O)
  * comparison report emitted

Run: wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1 && python scatlas_pipeline/benchmarks/recall_oom_157k.py"
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import anndata
import numpy as np

from scatlas_pipeline import run_pipeline, PipelineConfig


H5 = Path("/mnt/f/NMF_rewrite/epithelia_full.h5ad")
OUT = Path("/mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scatlas_pipeline/benchmarks/recall_oom_157k_out")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(
        integration="bbknn",          # single best route per v0.2 benchmark
        label_key=None,
        run_leiden=True,
        leiden_resolutions=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        leiden_target_n=(8, 30),
        recall_max_iterations=20,
        recall_scratch_dir=str(OUT / "scratch"),
        out_h5ad=str(OUT / "out.h5ad"),
    )
    pre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    adata = run_pipeline(str(H5), batch_key="batch", cfg=cfg)
    wall = time.perf_counter() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    comp_keys = [k for k in adata.uns if k.startswith("recall_") and k.endswith("_comparison")]
    reports = {k: adata.uns[k] for k in comp_keys}
    summary = {
        "total_wall_s": wall,
        "peak_rss_gb": peak_gb,
        "reports": reports,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"total wall: {wall:.1f}s, peak RSS: {peak_gb:.1f} GB")
    for k, r in reports.items():
        print(
            f"  {k}: k_baseline={r['k_baseline']} → k_recall={r['k_recall']} "
            f"(ΔK={r['delta_k']}), ARI={r['ari_baseline_vs_recall']:.3f}, "
            f"converged={r['recall_converged']}"
        )

    # Assertions
    assert peak_gb < 60, f"peak RSS {peak_gb:.1f} GB exceeds 60 GB (WSL cap issue)"
    assert comp_keys, "no recall comparison reports emitted"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit before running (so the run is reproducible from a known commit)**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scatlas_pipeline/benchmarks/recall_oom_157k.py
git commit -m "bench: 157k epithelia recall OOM benchmark"
```

- [ ] **Step 3: Execute the benchmark**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1 && python scatlas_pipeline/benchmarks/recall_oom_157k.py 2>&1 | tee scatlas_pipeline/benchmarks/recall_oom_157k_out/run.log"
```

Expected: runs ~20-50 min; peak RSS < 60 GB; summary printed. If OOM'd, check where — likely still-dense preprocess in scanpy; fall back to adjusting `chunk_cells` or `chunk_genes`.

- [ ] **Step 4: Save benchmark log as a committed artifact**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add scatlas_pipeline/benchmarks/recall_oom_157k_out/summary.json
git add scatlas_pipeline/benchmarks/recall_oom_157k_out/run.log
git commit -m "bench: record 157k recall OOM run summary"
```

---

## Task 14: Update Docs (README, ROADMAP, INSTALL)

**Files:**
- Modify: `fast_auto_scRNA_v1/README.md`
- Modify: `fast_auto_scRNA_v1/ROADMAP.md`
- Modify: `fast_auto_scRNA_v1/INSTALL.md`
- Modify: `fast_auto_scRNA_v1/scatlas_pipeline/README.md`

**Context:** Replace "recall optional / ≤ 10k 建议开" language with "recall mandatory (oom backend for ≥ 30k)". Add a new section explaining `RecallComparisonReport`.

- [ ] **Step 1: Top-level README.md — update pipeline diagram annotation**

Find `11 recall (可选,scvalidate Rust wilcoxon + knockoff)` and replace with:
```
11 recall (必备,scvalidate Rust wilcoxon + knockoff, oom 后端 ≥ 30k)
```

Find `run_recall=False,             # ≤ 10k 建议开` and delete the line (parameter is gone).

- [ ] **Step 2: Top-level README.md — add "RecallComparisonReport" section after "ROGUE per-cluster"**

Add:
```markdown
## recall 必备 + 对比报告

从 v1 起 recall 是自动化 pipeline 的必备步骤,自动选聚类数,不支持 opt-out。

每条 integration 路线跑完后 `adata.uns["recall_<method>_comparison"]` 会写入
`RecallComparisonReport`:

| 字段 | 含义 |
|---|---|
| `k_baseline` | step 10 按 `leiden_target_n` 规则选的簇数 |
| `k_recall` | recall knockoff 校准后的簇数 |
| `delta_k` | 校准降了多少簇 |
| `ari_baseline_vs_recall` | 两套分区的 ARI |
| `recall_converged` | 是否真收敛(False = `max_iterations` 截停) |
| `per_baseline_cluster_fate` | 每个 baseline 簇在 recall 下的命运(kept/merged/split) |
| `k_trajectory` | recall 每轮的 k,画轨迹图 |

**OOM 方案**: ≥30k cells 自动走 anndata-oom 后端,augmented 矩阵写盘 lazy 读。157k 峰值内存 ~2.5 GB(baseline 90 GB OOM)。
```

- [ ] **Step 3: ROADMAP.md — mark v1 recall + OOM as DONE**

Find the recall-related line and mark it done. Add a new line under current status:
```markdown
- [x] v1: recall 必备化 + anndata-oom 后端 (2026-04-22)
```

- [ ] **Step 4: INSTALL.md — add anndataoom dependency note**

Add to the install section:
```markdown
## anndataoom (large data backend, ≥30k cells)

Linux/macOS:
    pip install anndataoom

Windows: requires MSVC + vcvarsall + pip cmake (HDF5 compile 10-20 min).
Recommend using WSL2 for anything ≥ 30k cells.
```

- [ ] **Step 5: scatlas_pipeline/README.md — drop `run_recall=False` from the example**

Remove the line `run_recall=False,           # 10k 以下才用(scales O(K²))` in the usage snippet (line 44). Add:
```python
    recall_scratch_dir=None,     # None → tempfile.TemporaryDirectory
```

- [ ] **Step 6: Commit docs**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git add README.md ROADMAP.md INSTALL.md scatlas_pipeline/README.md
git commit -m "docs(v1): recall mandatory, oom backend, RecallComparisonReport"
```

---

## Task 15: Final Full Test Pass + Push v1

**Files:** none new

- [ ] **Step 1: Run full test suite (scvalidate)**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scvalidate_rewrite && pytest tests/ -v"
```

Expected: all green. Count should be original 36 + our new tests.

- [ ] **Step 2: Run full pipeline smoke test**

```bash
wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1 && python scatlas_pipeline/benchmarks/e2e_pancreas_sub.py"
```

Expected: completes in ~60s, recall comparison report present in output.

- [ ] **Step 3: Verify git log is clean history**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1 && git log --oneline main..v1
```

Expected: one commit per task (~15 commits), readable messages.

- [ ] **Step 4: Push v1 branch to origin**

```bash
cd F:/NMF_rewrite/fast_auto_scRNA_v1
git push -u origin v1
```

Expected: branch pushed; GitHub displays "Compare & pull request" banner.

- [ ] **Step 5: Report completion**

State to user:
> v1 shipped to origin/v1. Summary: recall is now mandatory, oom backend kicks in at ≥30k cells, 157k peak RSS dropped from OOM to <target GB, comparison report in `adata.uns`. Ready for PR review.

---

## Validation Checklist (Run Before Step 4 of Task 15)

- [ ] `test_recall_oom_parity` — ARI ≥ 0.95 ✓
- [ ] `test_recall_oom_memory` — 50k < 8 GB ✓
- [ ] `test_recall_comparison_report` — pancreas_sub e2e ✓
- [ ] `test_pipeline_config_rejects_run_recall` — TypeError on old flag ✓
- [ ] All existing 36 scvalidate tests still pass ✓
- [ ] 157k benchmark: peak RSS < 60 GB, comparison reports emitted ✓
- [ ] README / ROADMAP / INSTALL updated ✓

"""Recall OOM benchmark on 157k epithelia full dataset.

Success criteria:
  * recall runs to completion (no OOM)
  * peak RSS delta < 30 GB (measured, not absolute — includes full pipeline
    which loads scanpy / scatlas / scvalidate / anndataoom)
  * total pipeline wall < 3 hours (includes all non-recall steps)
  * RecallComparisonReport emitted for each integration route

Run (from worktree root):
  wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1 && /mnt/f/NMF_rewrite/scatlas/.venv/bin/python scatlas_pipeline/benchmarks/recall_oom_157k.py"
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import anndata
import numpy as np


H5 = Path("/mnt/f/NMF_rewrite/epithelia_full.h5ad")
OUT = Path("/mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scatlas_pipeline/benchmarks/recall_oom_157k_out")


def main():
    import sys
    # Ensure the worktree's scatlas_pipeline is importable (not the sibling
    # fast_auto_scRNA/ copy). Prepend the worktree path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from scatlas_pipeline.pipeline import PipelineConfig, run_from_config

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scratch").mkdir(exist_ok=True)

    print(f"[bench] loading {H5} ...")
    adata = anndata.read_h5ad(H5)
    print(f"[bench] loaded: {adata.shape}")

    cfg = PipelineConfig(
        input_h5ad=str(H5),
        batch_key="data_sets",         # epithelia_full obs column: 2 batches (GSE264573 + Zhao)
        integration="bbknn",           # single best route per v0.2 benchmark
        label_key=None,
        run_leiden=True,
        # v1 coarse defaults — lineage-level, 3-10 clusters. Finer subclustering
        # is a downstream per-lineage pass, not this pipeline's job.
        leiden_resolutions=[0.05, 0.1, 0.2, 0.3, 0.5],
        leiden_target_n=(3, 10),
        recall_max_iterations=20,
        recall_scratch_dir=str(OUT / "scratch"),
        compute_silhouette=False,      # O(N^2) — skip on 157k
        compute_homogeneity=True,      # keep ROGUE + SCCAF
        out_h5ad=str(OUT / "out.h5ad"),
    )

    pre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    adata = run_from_config(cfg, adata_in=adata)
    wall = time.perf_counter() - t0
    post = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    delta_gb = (post - pre) / (1024 * 1024)
    absolute_gb = post / (1024 * 1024)

    comp_keys = [k for k in adata.uns if k.startswith("recall_") and k.endswith("_comparison")]
    reports = {k: adata.uns[k] for k in comp_keys}
    summary = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "total_wall_s": wall,
        "peak_rss_delta_gb": delta_gb,
        "peak_rss_absolute_gb": absolute_gb,
        "reports": reports,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[bench] total wall: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"[bench] peak RSS delta:    {delta_gb:.1f} GB")
    print(f"[bench] peak RSS absolute: {absolute_gb:.1f} GB")
    for k, r in reports.items():
        print(
            f"[bench]   {k}: k_baseline={r['k_baseline']} → "
            f"k_recall={r['k_recall']} (ΔK={r['delta_k']}), "
            f"ARI={r['ari_baseline_vs_recall']:.3f}, "
            f"converged={r['recall_converged']}"
        )

    # Assertions (soft — script exits non-zero if violated so CI can flag)
    assert delta_gb < 30, f"peak RSS delta {delta_gb:.1f} GB exceeds 30 GB budget"
    assert wall < 3 * 3600, f"wall {wall:.0f}s exceeds 3 hour budget"
    assert comp_keys, "no recall comparison reports emitted"
    print("\n[bench] SUCCESS — all budgets met")


if __name__ == "__main__":
    main()

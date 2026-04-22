"""Recall OOM benchmark on 222k prostate all-cells dataset.

StepF.All_Cells.h5ad: 222,529 cells × 20,055 genes × 10 data.sets batches.
Has ground-truth annotation at three granularities:
  ct.main (3 levels: Immune / Epithelia / Stromal)
  ct.sub (7 levels: Epithelia / T / Mye / B / Endo / Fib / Mast)
  ct.sub.epi (13 levels: fine epithelia subtypes)

Success criteria:
  * recall runs to completion without OOM
  * peak RSS delta < 40 GB (222k is larger than 157k; budget bumped)
  * total wall < 3 hours
  * RecallComparisonReport emitted
  * (informational) ARI of recall vs ct.main reported for ground-truth
    validation — the pipeline itself doesn't depend on labels.

Run:
  wsl bash -c "cd /mnt/f/NMF_rewrite/fast_auto_scRNA_v1 && \\
      /mnt/f/NMF_rewrite/scatlas/.venv/bin/python \\
      scatlas_pipeline/benchmarks/recall_oom_222k_prostate.py"
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import anndata
import numpy as np


H5 = Path("/mnt/f/NMF_rewrite/StepF.All_Cells.h5ad")
OUT = Path(
    "/mnt/f/NMF_rewrite/fast_auto_scRNA_v1/scatlas_pipeline/benchmarks/"
    "recall_oom_222k_out"
)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from scatlas_pipeline.pipeline import PipelineConfig, run_from_config

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scratch").mkdir(exist_ok=True)

    print(f"[bench] loading {H5} ...")
    adata = anndata.read_h5ad(H5)
    print(f"[bench] loaded: {adata.shape}; obs cols: {list(adata.obs.columns)}")

    # Surface ground-truth class distributions up-front for later ARI context.
    for col in ("ct.main", "ct.sub"):
        if col in adata.obs.columns:
            vc = adata.obs[col].value_counts()
            print(f"[bench]   {col}: {dict(vc.head(10))}")

    cfg = PipelineConfig(
        input_h5ad=str(H5),
        batch_key="data.sets",         # 10 papers/batches (Chen 2021 Batch1/2/3, Kfoury, etc.)
        integration="bbknn",
        label_key="ct.sub",            # 7-class ground truth for scIB cLISI/silhouette
        # seurat_v3 HVG's scikit-misc loess segfaulted on 222k×10-batch
        # silently (exit 1, no traceback, mem < 1 GB). Use seurat HVG
        # instead — Gaussian-model HVG, no C-library dependency.
        hvg_flavor="seurat",
        hvg_batch_aware=False,
        run_leiden=True,
        leiden_resolutions=[0.05, 0.1, 0.2, 0.3, 0.5],
        leiden_target_n=(3, 10),
        recall_max_iterations=20,
        recall_scratch_dir=str(OUT / "scratch"),
        compute_silhouette=False,      # O(N^2) — skip at 222k
        compute_homogeneity=False,     # SCCAF LR CV on 222k × 7-class was
                                       # exceeding 1h in prior run; skip
                                       # here to surface recall numbers.
                                       # ROGUE already surfaced inside the
                                       # pipeline log even when False.
        out_h5ad=str(OUT / "out.h5ad"),
    )

    pre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    adata = run_from_config(cfg, adata_in=adata)
    wall = time.perf_counter() - t0
    post = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    delta_gb = (post - pre) / (1024 * 1024)
    absolute_gb = post / (1024 * 1024)

    comp_keys = [
        k for k in adata.uns
        if k.startswith("recall_") and k.endswith("_comparison")
    ]
    reports = {k: adata.uns[k] for k in comp_keys}

    # Ground-truth ARI for recall labels (informational — pipeline does not
    # use labels internally, but we have ct.main/ct.sub so quantify fit).
    from sklearn.metrics import adjusted_rand_score
    gt_ari = {}
    for method in ("bbknn",):
        recall_key = f"recall_{method}"
        if recall_key in adata.obs.columns:
            for gt_col in ("ct.main", "ct.sub", "ct.sub.epi"):
                if gt_col in adata.obs.columns:
                    gt_ari[f"{recall_key}__{gt_col}"] = float(
                        adjusted_rand_score(
                            adata.obs[gt_col].astype(str).to_numpy(),
                            adata.obs[recall_key].astype(str).to_numpy(),
                        )
                    )

    summary = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "total_wall_s": wall,
        "peak_rss_delta_gb": delta_gb,
        "peak_rss_absolute_gb": absolute_gb,
        "reports": reports,
        "gt_ari_vs_recall": gt_ari,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[bench] total wall: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"[bench] peak RSS delta:    {delta_gb:.1f} GB")
    print(f"[bench] peak RSS absolute: {absolute_gb:.1f} GB")
    for k, r in reports.items():
        print(
            f"[bench]   {k}: k_baseline={r['k_baseline']} → "
            f"k_recall={r['k_recall']} (ΔK={r['delta_k']}), "
            f"ARI(base,recall)={r['ari_baseline_vs_recall']:.3f}, "
            f"converged={r['recall_converged']}"
        )
    if gt_ari:
        print("\n[bench] ground-truth ARI (recall vs ct.*):")
        for k, v in gt_ari.items():
            print(f"[bench]   {k}: {v:.3f}")

    assert delta_gb < 40, f"peak RSS delta {delta_gb:.1f} GB exceeds 40 GB budget"
    assert wall < 3 * 3600, f"wall {wall:.0f}s exceeds 3 hour budget"
    assert comp_keys, "no recall comparison reports emitted"
    print("\n[bench] SUCCESS — all budgets met")


if __name__ == "__main__":
    main()

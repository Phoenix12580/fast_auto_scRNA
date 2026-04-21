"""Benchmark: full scatlas_pipeline end-to-end on 157k epithelia.

Exercises every Rust kernel + scanpy leiden + (optional) scvalidate recall.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scatlas_pipeline import PipelineConfig, run_from_config


def main() -> int:
    cfg = PipelineConfig(
        input_h5ad="/mnt/f/NMF_rewrite/epithelia_full.h5ad",
        batch_key="orig.ident",
        run_harmony=True,
        run_umap=True,
        run_leiden=True,
        run_recall=False,        # skip at 157k
        run_metrics=True,
        label_key="subtype",
        out_h5ad=None,           # don't write — just bench
    )
    run_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

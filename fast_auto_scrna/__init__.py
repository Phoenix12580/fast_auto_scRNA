"""fast_auto_scRNA — Rust-accelerated single-cell atlas pipeline.

Pipeline stages are organized as sibling subpackages (see README.md). The
main entry points are:

    from fast_auto_scrna import PipelineConfig, run_from_config

Stage modules can also be used standalone:

    from fast_auto_scrna.pca import pca
    from fast_auto_scrna.integration import bbknn, harmony
    from fast_auto_scrna.cluster import resolution   # graph-silhouette selector
"""
from __future__ import annotations

__version__ = "2.0.0.dev0"

# Re-exports — populated during V2-P2 migration.
# from .config import PipelineConfig
# from .runner import run_from_config

__all__ = [
    "__version__",
    # "PipelineConfig",
    # "run_from_config",
]

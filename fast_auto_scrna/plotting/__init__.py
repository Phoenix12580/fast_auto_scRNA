"""Cross-route comparison plots (UMAP grid, scIB heatmap, per-cluster ROGUE)."""
from .comparison import (
    compare_integration_plot,
    compare_scib_heatmap,
    compare_rogue_per_cluster,
    scib_comparison_table,
)

__all__ = [
    "compare_integration_plot",
    "compare_scib_heatmap",
    "compare_rogue_per_cluster",
    "scib_comparison_table",
]

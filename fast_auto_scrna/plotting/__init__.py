"""Per-route + cross-route diagnostic plots.

Per-route:
  * ``plot_route_umap``        — UMAP of this route colored by batch /
                                  GT / Leiden
  * ``plot_silhouette_curve``  — re-exported from cluster.resolution
  * ``compare_rogue_per_cluster`` with ``methods=(m,)`` — works for a
    single route too (one panel instead of a grid)
  * ``emit_route_plots``       — dispatcher, writes all of the above for
                                  one route into a plot dir

Cross-route:
  * ``compare_integration_plot``    — side-by-side UMAPs
  * ``compare_scib_heatmap``        — methods × metrics
  * ``compare_rogue_per_cluster``   — per-cluster ROGUE bar panels
  * ``scib_comparison_table``       — raw list-of-dicts table
"""
from .comparison import (
    compare_integration_plot,
    compare_scib_heatmap,
    compare_rogue_per_cluster,
    scib_comparison_table,
    plot_route_umap,
    emit_route_plots,
)
from ..cluster.resolution import (
    plot_silhouette_curve, plot_conductance_curve, plot_knee_curve,
)

__all__ = [
    "compare_integration_plot",
    "compare_scib_heatmap",
    "compare_rogue_per_cluster",
    "scib_comparison_table",
    "plot_route_umap",
    "plot_silhouette_curve",
    "plot_conductance_curve",
    "plot_knee_curve",
    "emit_route_plots",
]

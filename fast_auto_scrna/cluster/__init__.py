"""Stage 10 — Leiden clustering + knee-based resolution selector."""
from .leiden import leiden
from .resolution import (
    optimize_resolution_graph_silhouette,
    optimize_resolution_conductance,
    optimize_resolution_knee,
    mean_conductance,
    perpendicular_elbow,
    first_plateau_after_rise,
    pick_best_resolution,
    plot_silhouette_curve,
    plot_conductance_curve,
    plot_knee_curve,
    auto_resolution,
)

__all__ = [
    "leiden",
    "optimize_resolution_graph_silhouette",
    "optimize_resolution_conductance",
    "optimize_resolution_knee",
    "mean_conductance",
    "perpendicular_elbow",
    "first_plateau_after_rise",
    "pick_best_resolution",
    "plot_silhouette_curve",
    "plot_conductance_curve",
    "plot_knee_curve",
    "auto_resolution",
]

"""Stage 10 — Leiden clustering + CHAMP resolution selector (Weir 2017).

v2-P12: collapsed to CHAMP-only. The previous knee / conductance /
graph_silhouette / target_n optimizers were removed — see
``cluster/resolution.py`` module docstring for the rationale.
"""
from .leiden import leiden
from .resolution import (
    auto_resolution,
    mean_conductance,
    plot_champ_curve,
)
from .champ import optimize_resolution_champ

__all__ = [
    "leiden",
    "auto_resolution",
    "optimize_resolution_champ",
    "mean_conductance",
    "plot_champ_curve",
]

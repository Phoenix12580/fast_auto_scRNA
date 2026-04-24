"""Stage 06 — batch integration (BBKNN / Harmony 2 / none).

The three route modules are kept separate because each has a distinct
integration philosophy (graph-level, embedding-level, baseline).
``dispatch_integration(method, adata, cfg, ...)`` routes to the right one.
"""
from .bbknn import bbknn_kneighbors, bbknn
from .harmony import harmony

__all__ = ["bbknn_kneighbors", "bbknn", "harmony"]

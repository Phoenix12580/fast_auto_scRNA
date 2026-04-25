"""Stage 06 — batch integration routes.

Each route has a distinct integration philosophy:
  * ``bbknn`` — graph-level (modify kNN to be batch-balanced)
  * ``harmony`` — embedding-level (Harmony 2 PCA correction)
  * ``fastmnn`` — anchor-based (mutual nearest neighbors, Haghverdi 2018)
  * ``scvi`` — VAE-based (scvi-tools SCVI on raw counts)
"""
from .bbknn import bbknn_kneighbors, bbknn
from .harmony import harmony
from .fastmnn import fastmnn
from .scvi_route import scvi_train

__all__ = [
    "bbknn_kneighbors", "bbknn",
    "harmony",
    "fastmnn",
    "scvi_train",
]

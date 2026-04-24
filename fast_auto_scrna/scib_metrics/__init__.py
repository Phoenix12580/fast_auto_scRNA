"""Stage 08 — scIB metrics + cluster-homogeneity scores.

- ``scib`` — LISI (iLISI / cLISI), graph_connectivity, kBET, silhouettes
  (label / batch / isolated), composed via :func:`scib_score`.
- ``sccaf`` — logistic-regression CV accuracy on the embedding. sklearn
  fallback now; will dispatch to a Rust LR kernel when available.
"""
from .scib import (
    lisi,
    ilisi,
    clisi,
    graph_connectivity,
    kbet,
    scib_score,
)
from .sccaf import sccaf_accuracy

__all__ = [
    "lisi",
    "ilisi",
    "clisi",
    "graph_connectivity",
    "kbet",
    "scib_score",
    "sccaf_accuracy",
]

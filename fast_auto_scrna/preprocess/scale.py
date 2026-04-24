"""Stage 04 — z-score with clip. scanpy passthrough.

Planned Rust replacement will use SIMD-parallel column stats + in-place
normalization to dodge the dense-matrix OOM scanpy triggers on >500k ×
HVG matrices (OOM-1 milestone).
"""
from __future__ import annotations


def scale(
    adata,
    *,
    max_value: float | None = 10.0,
    zero_center: bool = True,
    layer: str | None = None,
    copy: bool = False,
):
    """Z-score each gene, clipping ``|z| ≤ max_value``. Standard PCA prep."""
    import scanpy as sc

    if copy:
        adata = adata.copy()
    sc.pp.scale(
        adata,
        max_value=max_value,
        zero_center=bool(zero_center),
        layer=layer,
    )
    return adata

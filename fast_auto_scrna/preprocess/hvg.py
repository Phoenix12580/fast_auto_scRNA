"""Stage 03 — highly-variable gene selection. scanpy passthrough.

A Rust kernel is planned but scanpy's is already ~25 s on 222k × 20k
and not the current bottleneck.
"""
from __future__ import annotations

from typing import Any


def highly_variable_genes(
    adata,
    *,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    batch_key: str | None = None,
    layer: str | None = None,
    subset: bool = False,
    inplace: bool = True,
) -> Any:
    """Pick HVGs, writing ``adata.var['highly_variable']``.

    Thin wrapper over :func:`scanpy.pp.highly_variable_genes`.

    Parameters
    ----------
    n_top_genes
        Number of HVGs to pick. 2000 is the scanpy/Seurat standard.
    flavor
        ``"seurat_v3"`` (default, VST on raw counts),
        ``"seurat"`` (dispersion on log1p),
        ``"cell_ranger"`` (10x Cell Ranger heuristic).
    batch_key
        If set, HVGs are selected per batch then intersected.
    layer
        If set, use ``adata.layers[layer]`` (e.g., raw counts for seurat_v3).
    """
    import scanpy as sc

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=int(n_top_genes),
        flavor=flavor,
        batch_key=batch_key,
        layer=layer,
        subset=bool(subset),
        inplace=bool(inplace),
    )
    return adata if inplace else None

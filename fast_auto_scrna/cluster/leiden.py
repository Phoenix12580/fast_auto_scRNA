"""Thin wrapper around scanpy's Leiden so every call site uses the same
defaults (igraph flavor, undirected, our chosen n_iterations)."""
from __future__ import annotations


def leiden(
    adata,
    *,
    resolution: float,
    key_added: str,
    adjacency=None,
    n_iterations: int = 2,
    random_state: int = 0,
):
    """Run Leiden, writing labels to ``adata.obs[key_added]``.

    If ``adjacency`` is given, pass it directly to sc.tl.leiden (bypasses
    scanpy's NeighborsView lookup). Otherwise scanpy uses
    ``adata.obsp['connectivities']`` per its convention.
    """
    import scanpy as sc

    kwargs = dict(
        resolution=resolution,
        key_added=key_added,
        flavor="igraph",
        directed=False,
        n_iterations=n_iterations,
        random_state=random_state,
    )
    if adjacency is not None:
        kwargs["adjacency"] = adjacency
    sc.tl.leiden(adata, **kwargs)

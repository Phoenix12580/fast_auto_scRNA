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

# scVI route imports torch via scvi-tools — heavy + GPU-only on Windows.
# Make it lazy: missing scvi-tools (e.g. WSL hosts that don't run scvi)
# should not break `from fast_auto_scrna.integration import ...`.
def scvi_train(*args, **kwargs):
    """Lazy proxy. Imports scvi_route on first call.

    Raises ImportError with install hint if scvi-tools is not installed.
    """
    try:
        from .scvi_route import scvi_train as _real
    except ImportError as e:
        raise ImportError(
            "scvi route requires scvi-tools. Install via:\n"
            '  pip install -e ".[scvi]"\n'
            "Note: scvi training is GPU-bound; install on the host with "
            "CUDA torch (typically Windows for this project)."
        ) from e
    return _real(*args, **kwargs)


__all__ = [
    "bbknn_kneighbors", "bbknn",
    "harmony",
    "fastmnn",
    "scvi_train",
]

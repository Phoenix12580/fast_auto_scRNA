"""anndata-oom backed path for find_clusters_recall."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata
from anndata.io import write_elem
import pandas as pd
import h5py


def _write_augmented_h5ad(
    real_gxc: np.ndarray,
    knock_gxc: np.ndarray,
    out_path: Path,
    chunk_cells: int = 5000,
) -> None:
    """Write [real | knockoff] stacked gene-wise to disk as AnnData cells × (2G).

    Streams N cells in chunks so RAM peak ≈ 2G × chunk_cells × 4B.
    Stored transposed to match scanpy's cells × genes convention.
    Single-pass write: h5py streams X; anndata.io.write_elem attaches obs/var
    without ever loading X back into memory.
    """
    if real_gxc.shape != knock_gxc.shape:
        raise ValueError(
            f"real_gxc and knock_gxc must have the same shape, got "
            f"{real_gxc.shape} vs {knock_gxc.shape}"
        )
    G, N = real_gxc.shape
    # Build var table labelling real/knockoff rows.
    # IMPORTANT: indices MUST be dtype='object' or pandas 2.x + pyarrow can
    # infer ArrowStringArray, which anndata.write_h5ad cannot serialize
    # (raises IORegistryError on write). Explicit dtype='object' below.
    var = pd.DataFrame(
        {"is_knockoff": [False] * G + [True] * G},
        index=pd.Index(
            [f"g{i}" for i in range(G)] + [f"k{i}" for i in range(G)],
            dtype="object",
        ),
    )
    obs = pd.DataFrame(
        index=pd.Index([f"c{i}" for i in range(N)], dtype="object"),
    )

    # Single-pass write: stream X via h5py, attach obs/var via write_elem.
    # No Pass 2 read — avoids materialising (N × 2G × 4B) in RAM.
    try:
        with h5py.File(out_path, "w") as f:
            # Root attrs required by the AnnData on-disk format
            f.attrs["encoding-type"] = "anndata"
            f.attrs["encoding-version"] = "0.1.0"

            # X: chunked streaming write
            dset = f.create_dataset(
                "X", shape=(N, 2 * G), dtype=np.float32,
                chunks=(min(chunk_cells, N), 2 * G),
                compression=None,
            )
            dset.attrs["encoding-type"] = "array"
            dset.attrs["encoding-version"] = "0.2.0"
            for start in range(0, N, chunk_cells):
                end = min(start + chunk_cells, N)
                chunk = np.empty((end - start, 2 * G), dtype=np.float32)
                chunk[:, :G] = real_gxc[:, start:end].T.astype(np.float32, copy=False)
                chunk[:, G:] = knock_gxc[:, start:end].T.astype(np.float32, copy=False)
                dset[start:end, :] = chunk

            # obs + var via anndata.io.write_elem (encodes proper AnnData metadata)
            write_elem(f, "obs", obs)
            write_elem(f, "var", var)
            # Empty auxiliary groups for AnnData format completeness
            write_elem(f, "uns", {})
            write_elem(f, "obsm", {})
            write_elem(f, "varm", {})
            write_elem(f, "obsp", {})
            write_elem(f, "varp", {})
            write_elem(f, "layers", {})
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise


def _wilcoxon_pair_chunked_dense(
    log_counts_gxc: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    chunk: int = 2000,
) -> np.ndarray:
    """Chunked per-gene Wilcoxon on a dense ndarray, for parity testing.

    Real production path uses _wilcoxon_pair_chunked_oom (reads h5ad chunks).
    Both call the same Rust kernel.
    """
    from scvalidate.recall_py.core import _wilcoxon_per_gene
    G = log_counts_gxc.shape[0]
    out = np.empty(G, dtype=np.float64)
    for start in range(0, G, chunk):
        end = min(start + chunk, G)
        out[start:end] = _wilcoxon_per_gene(
            log_counts_gxc[start:end], mask1, mask2,
        )
    return out


def _wilcoxon_pair_chunked_oom(
    log_counts_adata,         # backed AnnData (cells x 2G)
    mask1: np.ndarray,        # cell-level boolean
    mask2: np.ndarray,
    chunk_genes: int = 2000,
) -> np.ndarray:
    """Stream gene chunks from a backed AnnData, run Rust Wilcoxon, concat p."""
    from scvalidate.recall_py.core import _wilcoxon_per_gene
    n_vars = log_counts_adata.n_vars  # == 2G
    out = np.empty(n_vars, dtype=np.float64)
    for start in range(0, n_vars, chunk_genes):
        end = min(start + chunk_genes, n_vars)
        # Pull dense slab genes[start:end] for all cells (cells x chunk)
        # Convert to genes x cells for _wilcoxon_per_gene convention
        sl = log_counts_adata[:, start:end].X
        if hasattr(sl, "toarray"):
            sl = sl.toarray()
        sl_gxc = np.asarray(sl).T
        out[start:end] = _wilcoxon_per_gene(sl_gxc, mask1, mask2)
    return out


def find_clusters_recall_oom(counts_gxc, **kwargs):
    raise NotImplementedError("oom backend — filled in Task 5")

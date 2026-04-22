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


def find_clusters_recall_oom(counts_gxc, **kwargs):
    raise NotImplementedError("oom backend — filled in Task 5")

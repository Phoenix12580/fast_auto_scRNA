"""anndata-oom backed path for find_clusters_recall."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata
import pandas as pd


def _write_augmented_h5ad(
    real_gxc: np.ndarray,
    knock_gxc: np.ndarray,
    out_path: Path,
    chunk_cells: int = 5000,
) -> None:
    """Write [real | knockoff] stacked gene-wise to disk as AnnData cells × (2G).

    Streams N cells in chunks so RAM peak ≈ 2G × chunk_cells × 4B.
    Stored transposed to match scanpy's cells × genes convention.
    """
    assert real_gxc.shape == knock_gxc.shape
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

    # Two-pass write: (a) h5py streaming dataset for X, (b) anndata to attach
    # obs/var with proper AnnData v0.8+ metadata.
    import h5py
    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset(
            "X", shape=(N, 2 * G), dtype=np.float32,
            chunks=(min(chunk_cells, N), 2 * G),
            compression=None,
        )
        for start in range(0, N, chunk_cells):
            end = min(start + chunk_cells, N)
            chunk = np.empty((end - start, 2 * G), dtype=np.float32)
            chunk[:, :G] = real_gxc[:, start:end].T.astype(np.float32, copy=False)
            chunk[:, G:] = knock_gxc[:, start:end].T.astype(np.float32, copy=False)
            dset[start:end, :] = chunk
        # Minimal AnnData h5 skeleton so anndata.read_h5ad round-trips
        for grp in ("obs", "var", "uns", "obsm", "varm", "obsp", "varp", "layers"):
            f.create_group(grp)
    # Re-attach obs/var via anndata
    ad = anndata.read_h5ad(out_path)
    ad.obs = obs
    ad.var = var
    ad.write_h5ad(out_path)


def find_clusters_recall_oom(counts_gxc, **kwargs):
    raise NotImplementedError("oom backend — filled in Task 5")

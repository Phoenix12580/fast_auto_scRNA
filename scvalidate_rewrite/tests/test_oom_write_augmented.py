"""Unit test for _write_augmented_h5ad."""
import numpy as np
import anndata
from pathlib import Path


def test_write_augmented_roundtrip(tmp_path: Path):
    from scvalidate.recall_py._oom_backend import _write_augmented_h5ad

    rng = np.random.default_rng(0)
    G, N = 500, 2000
    real = rng.poisson(1.0, size=(G, N)).astype(np.int32)
    knock = rng.poisson(1.0, size=(G, N)).astype(np.int32)

    out = tmp_path / "aug.h5ad"
    _write_augmented_h5ad(real, knock, out, chunk_cells=500)

    ad = anndata.read_h5ad(out)
    # Stored as cells x (2G) to match anndata convention
    assert ad.shape == (N, 2 * G)
    # First G columns are real (transposed), last G are knockoff
    np.testing.assert_array_equal(ad.X[:, :G].T.astype(np.int32), real)
    np.testing.assert_array_equal(ad.X[:, G:].T.astype(np.int32), knock)
    assert "is_knockoff" in ad.var.columns
    assert ad.var["is_knockoff"].iloc[:G].eq(False).all()
    assert ad.var["is_knockoff"].iloc[G:].eq(True).all()

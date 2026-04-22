"""Unit test for _write_augmented_h5ad."""
import numpy as np
import anndata
import pytest
import warnings
from pathlib import Path


def _assert_roundtrip(out: Path, real: np.ndarray, knock: np.ndarray) -> None:
    """Helper: load h5ad and verify shape + data + metadata round-trip."""
    G, N = real.shape
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ad = anndata.read_h5ad(out)
    # Stored as cells x (2G) to match anndata convention
    assert ad.shape == (N, 2 * G)
    # First G columns are real (transposed), last G are knockoff
    np.testing.assert_array_equal(ad.X[:, :G].T.astype(np.int32), real)
    np.testing.assert_array_equal(ad.X[:, G:].T.astype(np.int32), knock)
    assert "is_knockoff" in ad.var.columns
    assert ad.var["is_knockoff"].iloc[:G].eq(False).all()
    assert ad.var["is_knockoff"].iloc[G:].eq(True).all()


@pytest.mark.parametrize("N", [2000, 2003])
def test_write_augmented_roundtrip(tmp_path: Path, N: int):
    """Round-trip test: N=2000 (4 clean chunks) and N=2003 (chunk boundary off-by-one)."""
    from scvalidate.recall_py._oom_backend import _write_augmented_h5ad

    rng = np.random.default_rng(0)
    G = 500
    real = rng.poisson(1.0, size=(G, N)).astype(np.int32)
    knock = rng.poisson(1.0, size=(G, N)).astype(np.int32)

    out = tmp_path / f"aug_N{N}.h5ad"
    _write_augmented_h5ad(real, knock, out, chunk_cells=500)

    _assert_roundtrip(out, real, knock)


def test_write_augmented_cleanup_on_failure(tmp_path: Path):
    """Partial write must not leave a file behind when an error occurs."""
    from scvalidate.recall_py._oom_backend import _write_augmented_h5ad
    import unittest.mock as mock

    rng = np.random.default_rng(1)
    G, N = 100, 200
    real = rng.poisson(1.0, size=(G, N)).astype(np.int32)
    knock = rng.poisson(1.0, size=(G, N)).astype(np.int32)

    out = tmp_path / "aug_fail.h5ad"

    # Patch write_elem to raise mid-write
    with mock.patch(
        "scvalidate.recall_py._oom_backend.write_elem",
        side_effect=RuntimeError("injected failure"),
    ):
        with pytest.raises(RuntimeError, match="injected failure"):
            _write_augmented_h5ad(real, knock, out, chunk_cells=100)

    assert not out.exists(), "Partial file must be cleaned up on failure"

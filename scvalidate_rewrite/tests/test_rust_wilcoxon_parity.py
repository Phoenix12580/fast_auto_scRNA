"""Parity tests for scvalidate_rust.wilcoxon_ranksum_matrix vs the pure-Py
reference from scvalidate.recall_py.core._wilcoxon_per_gene (which itself uses
scipy's rankdata).

Required: scvalidate_rust extension built + installed. Skips entirely when the
extension is not importable, so the CI matrix without Rust still passes.
"""
from __future__ import annotations

import numpy as np
import pytest

rust = pytest.importorskip("scvalidate_rust")

from scvalidate.recall_py.core import _wilcoxon_per_gene


def _rust_wilcoxon(log_counts, mask1, mask2):
    return np.asarray(
        rust.wilcoxon_ranksum_matrix(
            np.ascontiguousarray(log_counts, dtype=np.float64),
            np.ascontiguousarray(mask1, dtype=bool),
            np.ascontiguousarray(mask2, dtype=bool),
        )
    )


def _two_way_close(a, b, *, atol=1e-8, rtol=0.0):
    return np.allclose(a, b, atol=atol, rtol=rtol)


class TestParityBasic:
    def test_no_ties_well_separated_groups(self):
        rng = np.random.default_rng(0)
        g, n = 40, 200
        x = rng.normal(size=(g, n)).astype(np.float64)
        # Introduce a real effect on the first 5 rows: +2 in group 1
        mask1 = np.zeros(n, dtype=bool); mask1[:100] = True
        mask2 = ~mask1
        x[:5, mask1] += 2.0

        p_py = _wilcoxon_per_gene(x, mask1, mask2)
        p_rs = _rust_wilcoxon(x, mask1, mask2)

        assert p_py.shape == (g,) == p_rs.shape
        assert _two_way_close(p_py, p_rs, atol=1e-10), (
            f"max |Δp| = {np.max(np.abs(p_py - p_rs)):.3e}"
        )

    def test_ties_heavy_log1p_counts(self):
        """Log1p of sparse counts gives many zeros — stress-test tie handling."""
        rng = np.random.default_rng(1)
        g, n = 30, 400
        # Sparse Poisson counts then log1p
        counts = rng.poisson(lam=0.3, size=(g, n)).astype(np.float64)
        x = np.log1p(counts)
        mask1 = np.zeros(n, dtype=bool); mask1[: n // 2] = True
        mask2 = ~mask1

        p_py = _wilcoxon_per_gene(x, mask1, mask2)
        p_rs = _rust_wilcoxon(x, mask1, mask2)

        # scipy's rankdata(method='average') vs manual tie-averaging must be
        # bit-exact enough to be well within 1e-8 after all the downstream
        # normal-approx math.
        assert _two_way_close(p_py, p_rs, atol=1e-8), (
            f"max |Δp| = {np.max(np.abs(p_py - p_rs)):.3e}"
        )

    def test_single_group_empty_returns_ones(self):
        """Python path returns ones when either group is empty."""
        x = np.random.default_rng(2).normal(size=(10, 20)).astype(np.float64)
        mask1 = np.zeros(20, dtype=bool)
        mask2 = np.ones(20, dtype=bool)
        p_rs = _rust_wilcoxon(x, mask1, mask2)
        assert p_rs.shape == (10,)
        assert np.all(p_rs == 1.0)

    def test_p_is_clipped(self):
        """Extremely separated groups → p floor at 1e-300."""
        g, n = 3, 400
        x = np.zeros((g, n), dtype=np.float64)
        # Row 0: all group 1 cells = 1.0, all group 2 cells = 0.0
        # (huge effect, z huge, p underflow-able)
        mask1 = np.zeros(n, dtype=bool); mask1[: n // 2] = True
        mask2 = ~mask1
        x[0, mask1] = 100.0
        # Row 1: flat (no effect, p ≈ 1)
        # Row 2: anti-effect
        x[2, mask2] = 100.0

        p_rs = _rust_wilcoxon(x, mask1, mask2)
        assert 0.95 < p_rs[1] <= 1.0
        # Rows 0 & 2: effect strong enough to hit floor or very close
        assert p_rs[0] <= 1e-40
        assert p_rs[2] <= 1e-40


class TestParityRecallShape:
    """Emulate the shape actually seen inside find_clusters_recall: augmented
    log matrix with 2G rows, masks picking two clusters out of k."""

    @pytest.mark.parametrize("n_cells", [100, 1000])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_recall_style_augmented(self, n_cells, seed):
        rng = np.random.default_rng(seed)
        g = 50            # real genes
        x_real = rng.poisson(lam=0.5, size=(g, n_cells))
        x_knock = rng.poisson(lam=0.5, size=(g, n_cells))
        aug = np.concatenate([x_real, x_knock], axis=0).astype(np.float64)
        # log1p-normalize in the same style as recall
        col_sums = aug.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        aug = np.log1p(aug * 1e4 / col_sums)

        # "Clusters": 3 equal partitions of cells; test pair (0, 1)
        third = n_cells // 3
        mask1 = np.zeros(n_cells, dtype=bool); mask1[:third] = True
        mask2 = np.zeros(n_cells, dtype=bool); mask2[third : 2 * third] = True

        p_py = _wilcoxon_per_gene(aug, mask1, mask2)
        p_rs = _rust_wilcoxon(aug, mask1, mask2)

        assert p_py.shape == (2 * g,) == p_rs.shape
        assert _two_way_close(p_py, p_rs, atol=1e-8), (
            f"n_cells={n_cells} seed={seed}: "
            f"max |Δp| = {np.max(np.abs(p_py - p_rs)):.3e}"
        )

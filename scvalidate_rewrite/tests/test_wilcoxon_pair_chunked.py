"""_wilcoxon_pair_chunked must match _wilcoxon_per_gene bit-close on dense input."""
import numpy as np


def test_chunked_matches_dense():
    from scvalidate.recall_py._oom_backend import _wilcoxon_pair_chunked_dense
    from scvalidate.recall_py.core import _wilcoxon_per_gene

    rng = np.random.default_rng(42)
    G, N = 1000, 500
    log_counts = rng.poisson(1.0, size=(G, N)).astype(np.float32)
    log_counts = np.log1p(log_counts)
    mask1 = rng.random(N) < 0.4
    mask2 = ~mask1

    p_full = _wilcoxon_per_gene(log_counts, mask1, mask2)
    p_chunked = _wilcoxon_pair_chunked_dense(log_counts, mask1, mask2, chunk=250)

    np.testing.assert_allclose(p_full, p_chunked, atol=1e-10)

"""Memory smoke test: oom backend at 50k cells peak RSS delta < 8 GB.

Reference (from v0.4 progress memory):
    dense 50k f32 peak RSS = 34 GB (absolute)
    oom target             = delta < 8 GB  (>= 4x reduction)

Why delta (post - pre), not absolute post: pytest pre-loads scanpy/anndata/
rdata/etc. which consume 4-5 GB before the test even starts. Absolute RSS
would reflect those imports rather than recall's own memory footprint. Delta
is the memory added *by* find_clusters_recall; that's the OOM budget we
actually care about.
"""
import resource
import pytest
import numpy as np


@pytest.mark.slow
def test_oom_memory_50k(tmp_path):
    from scvalidate.recall_py import find_clusters_recall
    rng = np.random.default_rng(0)
    G, N = 8000, 50_000
    counts = rng.poisson(0.3, size=(G, N)).astype(np.int32)

    # ru_maxrss on Linux is kB and reports lifetime peak — for the delta
    # metric we snapshot before/after and take the difference, which works
    # because only recall runs between the two calls.
    pre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    _ = find_clusters_recall(
        counts,
        resolution_start=0.8, max_iterations=3,
        fdr=0.05, seed=0, verbose=False,
        backend="oom", scratch_dir=tmp_path,
    )

    post = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    delta_gb = (post - pre) / (1024 * 1024)
    absolute_gb = post / (1024 * 1024)
    print(
        f"oom backend 50k: delta={delta_gb:.2f} GB, absolute={absolute_gb:.2f} GB "
        f"(dense baseline 34 GB absolute per memory)"
    )
    assert delta_gb < 8.0, (
        f"oom backend 50k peak delta {delta_gb:.2f} GB exceeds 8 GB budget "
        f"(absolute peak {absolute_gb:.2f} GB)"
    )

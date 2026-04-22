import numpy as np


def test_comparison_report_basic():
    from scvalidate.recall_py.comparison import build_comparison_report

    rng = np.random.default_rng(0)
    # Baseline has 10 clusters, recall merged it to 6
    labels_baseline = rng.integers(0, 10, size=1000)
    # recall: first 5 clusters kept, last 5 all merged into cluster 5
    labels_recall = np.where(labels_baseline < 5, labels_baseline, 5)

    rep = build_comparison_report(
        labels_baseline=labels_baseline,
        labels_recall=labels_recall,
        resolution_baseline=0.8,
        resolution_recall=0.4,
        recall_converged=True,
        k_trajectory=[10, 8, 6],
        recall_wall_time_s=123.4,
    )
    assert rep.k_baseline == 10
    assert rep.k_recall == 6
    assert rep.delta_k == 4
    assert 0.3 <= rep.ari_baseline_vs_recall <= 0.8
    # clusters 0..4 kept, 5..9 merged
    fates = rep.per_baseline_cluster_fate
    assert all("kept" in fates[c] for c in range(5))
    assert all("merged" in fates[c] for c in range(5, 10))

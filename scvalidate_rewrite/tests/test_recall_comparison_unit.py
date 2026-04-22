import numpy as np
import pytest


def test_pipeline_config_run_recall_defaults_off():
    """run_recall was temporarily reverted from 'deleted/mandatory' to
    'opt-in default off' while atlas-scale recall performance is reworked.
    The v1 spec intent is still to make recall mandatory — revisit once
    the perf rework lands."""
    import sys, os
    _pipeline_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if _pipeline_root not in sys.path:
        sys.path.insert(0, _pipeline_root)
    from scatlas_pipeline.pipeline import PipelineConfig
    cfg = PipelineConfig(input_h5ad="dummy", batch_key="b")
    assert cfg.run_recall is False, "recall must default off during rework"
    cfg2 = PipelineConfig(input_h5ad="dummy", batch_key="b", run_recall=True)
    assert cfg2.run_recall is True, "opt-in must still work"


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
    # clusters 0..4 kept, 5..9 merged. Keys are stringified ints for
    # h5ad-serializability (anndata.write_h5ad requires string dict keys).
    fates = rep.per_baseline_cluster_fate
    assert all(isinstance(k, str) for k in fates.keys()), (
        f"fate keys must be str, got types: {set(type(k) for k in fates.keys())}"
    )
    assert all("kept" in fates[str(c)] for c in range(5))
    assert all("merged" in fates[str(c)] for c in range(5, 10))

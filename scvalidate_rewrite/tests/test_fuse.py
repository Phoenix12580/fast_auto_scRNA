"""Unit tests for fuse.report."""

import pytest

from scvalidate.fuse import fuse_report


def test_gate_rejects_on_recall_fail():
    r = fuse_report(
        [0, 1],
        recall_pass={0: False, 1: True},
        scshc_pvalue={0: 0.001, 1: 0.001},
        rogue_score={0: 0.9, 1: 0.9},
        n_markers={0: 50, 1: 50},
        alpha=0.05,
    )
    verdicts = {c.cluster_id: c.verdict for c in r.per_cluster}
    assert verdicts[0] == "REJECT"
    assert verdicts[1] == "HIGH"


def test_gate_rejects_on_scshc_nonsig():
    r = fuse_report(
        [0, 1],
        recall_pass={0: True, 1: True},
        scshc_pvalue={0: 0.20, 1: 0.001},
        rogue_score={0: 0.9, 1: 0.9},
        n_markers={0: 50, 1: 50},
        alpha=0.05,
        enable_scshc=True,
    )
    verdicts = {c.cluster_id: c.verdict for c in r.per_cluster}
    assert verdicts[0] == "REJECT"
    assert verdicts[1] == "HIGH"


def test_scshc_disabled_by_default():
    """With enable_scshc=False (default), a non-sig scSHC p-value is ignored."""
    r = fuse_report(
        [0],
        recall_pass={0: True},
        scshc_pvalue={0: 0.99},
        rogue_score={0: 0.9},
        n_markers={0: 50},
        alpha=0.05,
    )
    c = r.per_cluster[0]
    assert c.verdict == "HIGH"
    assert c.scshc_pvalue is None


def test_bio_thresholds():
    # Cluster 0: bio score ~= 0.6*0.9 + 0.4*1.0 = 0.94 → HIGH
    # Cluster 1: bio score ~= 0.6*0.5 + 0.4*0.33 = 0.432 → MED
    # Cluster 2: bio score ~= 0.6*0.1 + 0.4*0.1 = 0.1 → LOW
    r = fuse_report(
        [0, 1, 2],
        recall_pass={0: True, 1: True, 2: True},
        scshc_pvalue={0: 0.001, 1: 0.001, 2: 0.001},
        rogue_score={0: 0.9, 1: 0.5, 2: 0.1},
        n_markers={0: 50, 1: 10, 2: 3},
        alpha=0.05,
    )
    verdicts = {c.cluster_id: c.verdict for c in r.per_cluster}
    assert verdicts[0] == "HIGH"
    assert verdicts[1] == "MED"
    assert verdicts[2] == "LOW"


def test_missing_signals_redistribute_weight():
    """ROGUE-only input should still give a verdict using the remaining signal."""
    r = fuse_report(
        [0],
        recall_pass={0: True},
        scshc_pvalue={0: 0.001},
        rogue_score={0: 0.9},
        # n_markers omitted
        alpha=0.05,
    )
    c = r.per_cluster[0]
    assert c.bio_score == pytest.approx(0.9, rel=1e-6)
    assert c.verdict == "HIGH"


def test_dataframe_output_columns():
    r = fuse_report(
        [0],
        recall_pass={0: True},
        scshc_pvalue={0: 0.001},
        rogue_score={0: 0.9},
        n_markers={0: 50},
    )
    df = r.to_dataframe()
    assert set(df.columns) == {
        "cluster_id",
        "recall_pass",
        "scshc_pvalue",
        "rogue",
        "n_markers",
        "bio_score",
        "verdict",
        "reason",
    }

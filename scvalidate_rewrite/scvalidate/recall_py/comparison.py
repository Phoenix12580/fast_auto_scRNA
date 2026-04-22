"""RecallComparisonReport — baseline Leiden k vs recall-calibrated k."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from collections import Counter
import numpy as np
from sklearn.metrics import adjusted_rand_score


@dataclass
class RecallComparisonReport:
    k_baseline: int
    k_recall: int
    resolution_baseline: float
    resolution_recall: float
    delta_k: int
    ari_baseline_vs_recall: float
    recall_converged: bool
    per_baseline_cluster_fate: dict            # {cluster_id: "kept"|"merged_with_{X}"|"split"}
    k_trajectory: list                          # recall per-iter k
    recall_wall_time_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_fate(labels_baseline: np.ndarray, labels_recall: np.ndarray) -> dict:
    """For each baseline cluster, describe what recall did to it.

    'kept'             — >= 80% of baseline cells stayed in one recall cluster
                         that nobody else dominates.
    'merged_with_{X}'  — >= 80% moved to a recall cluster that also absorbs
                         other baseline clusters.
    'split_into_[X,Y]' — baseline cluster is split across >=2 recall clusters
                         each holding >= 20%.
    """
    fates: dict = {}
    recall_dominance = {}  # recall_cluster -> set of baseline clusters it dominates
    for b in np.unique(labels_baseline):
        mask = labels_baseline == b
        counter = Counter(labels_recall[mask])
        total = mask.sum()
        top_recall, top_n = counter.most_common(1)[0]
        if top_n / total >= 0.80:
            recall_dominance.setdefault(int(top_recall), set()).add(int(b))
        else:
            # splits: recall clusters with >= 20% share
            splits = [int(r) for r, n in counter.items() if n / total >= 0.20]
            fates[str(int(b))] = f"split_into_{sorted(splits)}"

    for r, bs in recall_dominance.items():
        if len(bs) == 1:
            b = next(iter(bs))
            fates[str(b)] = "kept"
        else:
            for b in bs:
                others = sorted(bs - {b})
                fates[str(b)] = f"merged_with_{others}"
    return fates


def build_comparison_report(
    *,
    labels_baseline: np.ndarray,
    labels_recall: np.ndarray,
    resolution_baseline: float,
    resolution_recall: float,
    recall_converged: bool,
    k_trajectory: list,
    recall_wall_time_s: float,
) -> RecallComparisonReport:
    k_b = int(len(np.unique(labels_baseline)))
    k_r = int(len(np.unique(labels_recall)))
    ari = float(adjusted_rand_score(labels_baseline, labels_recall))
    fates = _classify_fate(labels_baseline, labels_recall)
    return RecallComparisonReport(
        k_baseline=k_b,
        k_recall=k_r,
        resolution_baseline=resolution_baseline,
        resolution_recall=resolution_recall,
        delta_k=k_b - k_r,
        ari_baseline_vs_recall=ari,
        recall_converged=recall_converged,
        per_baseline_cluster_fate=fates,
        k_trajectory=list(k_trajectory),
        recall_wall_time_s=recall_wall_time_s,
    )

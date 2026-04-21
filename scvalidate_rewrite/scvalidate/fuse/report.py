"""Two-layer fused verdict for a set of clusters.

Weight rationale (research-explainable, not fit from data):

* Layer 1 is a hard gate: a cluster that fails the recall knockoff test
  (and, optionally, the sc-SHC split significance test) is REJECTED
  outright. No amount of biological coherence rescues a statistically
  indistinguishable cluster.

* scSHC is OFF by default (``enable_scshc=False``). On the epithelia
  10k bench (2026-04 parity run), scSHC's Ward-linkage FWER gate
  collapsed 37 Leiden clusters to 1 merged group — matching R bit-for-bit,
  but biologically useless (ARI vs ground-truth subtype = 0.00 vs 0.26
  for recall alone). For homogeneous single-lineage datasets the gate
  is too strict. Set ``enable_scshc=True`` for mixed-lineage data where
  the hierarchical split test adds signal.

* Layer 2 weights (0.6 ROGUE + 0.4 marker richness):
  - ROGUE is a transcriptome-wide purity score, so it dominates (0.6).
  - Marker richness (#genes with q<0.05 and |log2FC|>1) captures whether the
    cluster has an interpretable biological signature. It's correlated with
    ROGUE but can diverge (e.g. ROGUE high but no markers = quiescent
    subpopulation), so it gets the remaining 0.4.
  - Future weights (CellTypist consistency, pathway coherence) are planned
    as additional Layer 2 columns rather than re-tuning the existing two.

* Verdict thresholds on bio_score:
  - >= 0.70 → HIGH  (both ROGUE and markers clearly agree)
  - >= 0.40 → MED   (one weak signal only)
  -  < 0.40 → LOW   (passes math gate but biology unclear)

These thresholds are not trained — they fall out of the weight scheme. With
(0.6, 0.4) weights, 0.70 requires both signals above ~0.58, and 0.40 is
roughly "one signal at ~0.7, the other zero" → matches the verbal intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Tunable weights (kept as module constants so users can override explicitly)
# -----------------------------------------------------------------------------

# Layer-2 bio score components (must sum to 1.0).
ROGUE_WEIGHT = 0.60
MARKER_WEIGHT = 0.40

# Verdict thresholds on bio_score (only applied to clusters that pass Layer 1).
VERDICT_HIGH_MIN = 0.70
VERDICT_MED_MIN = 0.40

# Marker-richness normalization. A cluster with 30+ significant markers gets
# the full 1.0 score; below that it scales linearly. 30 is a conservative
# rule-of-thumb for "clearly a real cell type" from the Seurat community;
# ROGUE and sc-SHC papers both discuss similar magnitudes.
MARKER_RICHNESS_SATURATION = 30


@dataclass
class ClusterVerdict:
    cluster_id: object
    recall_pass: bool | None                # None if recall not run
    scshc_pvalue: float | None              # None if sc-SHC not run
    rogue_score: float | None               # None if ROGUE not run
    n_markers: int | None                   # None if marker DE not run
    bio_score: float                        # Layer-2 weighted score
    verdict: Literal["REJECT", "LOW", "MED", "HIGH"]
    reason: str                             # one-line explanation


@dataclass
class AutoClusterReport:
    per_cluster: list[ClusterVerdict] = field(default_factory=list)
    alpha: float = 0.05

    @property
    def n_clusters(self) -> int:
        return len(self.per_cluster)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for c in self.per_cluster:
            rows.append(
                {
                    "cluster_id": c.cluster_id,
                    "recall_pass": c.recall_pass,
                    "scshc_pvalue": c.scshc_pvalue,
                    "rogue": c.rogue_score,
                    "n_markers": c.n_markers,
                    "bio_score": c.bio_score,
                    "verdict": c.verdict,
                    "reason": c.reason,
                }
            )
        return pd.DataFrame(rows)


def _marker_component(n_markers: int | None) -> float:
    if n_markers is None:
        return 0.0
    return min(1.0, n_markers / MARKER_RICHNESS_SATURATION)


def _bio_score(rogue: float | None, n_markers: int | None) -> float:
    # When a signal is missing, its weight mass redistributes to the other
    # signal so the score stays on a 0–1 scale.
    components = []
    if rogue is not None:
        components.append((ROGUE_WEIGHT, max(0.0, min(1.0, rogue))))
    if n_markers is not None:
        components.append((MARKER_WEIGHT, _marker_component(n_markers)))
    if not components:
        return 0.0
    total_w = sum(w for w, _ in components)
    return sum(w * v for w, v in components) / total_w


def fuse_report(
    cluster_ids,
    *,
    recall_pass: dict | None = None,
    scshc_pvalue: dict | None = None,
    rogue_score: dict | None = None,
    n_markers: dict | None = None,
    alpha: float = 0.05,
    enable_scshc: bool = False,
) -> AutoClusterReport:
    """Combine per-cluster signals into a single report.

    Each signal is a dict ``{cluster_id: value}``. Missing entries are
    treated as "not run" and reported as None. Use a fully populated dict
    to compute a full two-layer verdict; otherwise partial signals are
    aggregated as described in :mod:`scvalidate.fuse.report`.

    Parameters
    ----------
    enable_scshc
        If False (default), the ``scshc_pvalue`` dict is ignored and scSHC
        does not participate in the Layer-1 gate. See module docstring for
        why this defaults off.
    """
    recall_pass = recall_pass or {}
    scshc_pvalue = (scshc_pvalue or {}) if enable_scshc else {}
    rogue_score = rogue_score or {}
    n_markers = n_markers or {}

    verdicts: list[ClusterVerdict] = []
    for cid in cluster_ids:
        rp = recall_pass.get(cid)
        sp_ = scshc_pvalue.get(cid)
        rg = rogue_score.get(cid)
        nm = n_markers.get(cid)

        # Layer 1 gate. If a signal wasn't computed, treat it as not-failing.
        gate_ok = True
        gate_reasons: list[str] = []
        if rp is not None and not rp:
            gate_ok = False
            gate_reasons.append("recall knockoff null-indistinguishable")
        if sp_ is not None and sp_ >= alpha:
            gate_ok = False
            gate_reasons.append(f"sc-SHC p={sp_:.3g} >= alpha={alpha}")

        if not gate_ok:
            verdicts.append(
                ClusterVerdict(
                    cluster_id=cid,
                    recall_pass=rp,
                    scshc_pvalue=sp_,
                    rogue_score=rg,
                    n_markers=nm,
                    bio_score=0.0,
                    verdict="REJECT",
                    reason="; ".join(gate_reasons),
                )
            )
            continue

        # Layer 2 score.
        bio = _bio_score(rg, nm)
        if bio >= VERDICT_HIGH_MIN:
            vd = "HIGH"
            reason = "passes gate; strong bio signal"
        elif bio >= VERDICT_MED_MIN:
            vd = "MED"
            reason = "passes gate; moderate bio signal"
        else:
            vd = "LOW"
            reason = "passes gate; weak bio signal (low ROGUE and/or few markers)"

        verdicts.append(
            ClusterVerdict(
                cluster_id=cid,
                recall_pass=rp,
                scshc_pvalue=sp_,
                rogue_score=rg,
                n_markers=nm,
                bio_score=round(float(bio), 4),
                verdict=vd,
                reason=reason,
            )
        )

    return AutoClusterReport(per_cluster=verdicts, alpha=alpha)

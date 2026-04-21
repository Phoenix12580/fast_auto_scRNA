"""Two-layer cluster verdict fusion.

Layer 1 — math gate (hard): recall_pass AND scSHC_p < alpha
Layer 2 — bio score (soft 0–1): weighted sum of ROGUE purity + marker richness

See ``report.py`` for the verdict mapping and weight rationale.
"""

from scvalidate.fuse.report import AutoClusterReport, ClusterVerdict, fuse_report

__all__ = ["AutoClusterReport", "ClusterVerdict", "fuse_report"]

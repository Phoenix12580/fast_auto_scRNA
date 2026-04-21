"""scvalidate: automated single-cell clustering validation."""

from scvalidate.fuse import AutoClusterReport, fuse_report
from scvalidate.recall_py import RecallResult, find_clusters_recall
from scvalidate.rogue_py import calculate_rogue, entropy_table, se_fun
from scvalidate.scshc_py import scshc
from scvalidate.scshc_py import test_clusters as test_scshc_clusters

__version__ = "0.1.0.dev0"

__all__ = [
    "AutoClusterReport",
    "RecallResult",
    "calculate_rogue",
    "entropy_table",
    "find_clusters_recall",
    "fuse_report",
    "scshc",
    "se_fun",
    "test_scshc_clusters",
]

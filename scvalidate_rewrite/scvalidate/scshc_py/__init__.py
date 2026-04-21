"""sc-SHC: Significance of Hierarchical Clustering for Single-Cell Data.

Port of igrabski/sc-SHC (R, main).

Reference: Grabski, Street & Irizarry 2023, Nat Methods.
    "Significance analysis for clustering with single-cell RNA-sequencing data"

Core idea: hierarchical Ward clustering on Poisson-deviance residuals, then
FWER-controlled testing of every binary split via a LogNormal-Poisson
generative null fitted by method of moments.
"""

from scvalidate.scshc_py.core import (
    BatchParams,
    SCSHCNode,
    compute_ess,
    fit_model,
    fit_model_batch,
    poisson_dev_batch,
    poisson_dispersion_stats,
    reduce_dimension,
    scshc,
    test_clusters,
    ward_linkage_stat,
)

__all__ = [
    "BatchParams",
    "SCSHCNode",
    "compute_ess",
    "fit_model",
    "fit_model_batch",
    "poisson_dev_batch",
    "poisson_dispersion_stats",
    "reduce_dimension",
    "scshc",
    "test_clusters",
    "ward_linkage_stat",
]

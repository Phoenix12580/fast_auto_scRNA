"""Debug scSHC's root split on pancreas_sub.

Compute the observed Ward stat and 10 null Ward stats separately.
Then compare magnitudes — if observed << null, that's the bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import mmread

from scvalidate.scshc_py.core import (
    _deviance_feature_selection, _generate_null, _generate_null_statistic,
    _test_split, fit_model, poisson_dispersion_stats, reduce_dimension,
    ward_linkage_stat,
)
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import norm

BENCH = "F:/NMF_rewrite/scvalidate_rewrite/benchmark"

# Load
counts = mmread(f"{BENCH}/counts.mtx").tocsr()
cells = open(f"{BENCH}/cells.tsv").read().strip().split("\n")
cl_df = pd.read_csv(f"{BENCH}/scvalidate_clusters.csv", index_col=0).loc[cells]
leiden = cl_df["leiden"].astype(str).to_numpy()

data = np.asarray(counts.todense())  # G x N
print(f"data: {data.shape}, leiden: {len(np.unique(leiden))} clusters")

# Var_genes
dev = _deviance_feature_selection(data)
var_idx = np.argsort(-dev)[:2000]

# Build pseudobulk tree, get root 2-way cut
unique_ids = np.unique(leiden)
pb = np.zeros((len(unique_ids), len(var_idx)))
for i, cid in enumerate(unique_ids):
    pb[i] = data[np.ix_(var_idx, np.where(leiden == cid)[0])].sum(axis=1)
pb = pb / pb.sum(axis=1, keepdims=True).clip(min=1e-12)
pb_Z = linkage(pdist(pb), method="ward")
cuts = fcluster(pb_Z, t=2, criterion="maxclust")
left = unique_ids[cuts == 1]
right = unique_ids[cuts == 2]
print(f"Root split: left={list(left)}, right={list(right)}")

# TEST 2: clean biological pair — leiden 0 (biggest) vs leiden 1 (likely different CellType)
print("\n=== TEST 2: leiden 0 vs leiden 1 (clean binary) ===")
left = np.array(["0"])
right = np.array(["1"])

ids1 = np.where(np.isin(leiden, left))[0]
ids2 = np.where(np.isin(leiden, right))[0]
print(f"ids1: {len(ids1)} cells  ids2: {len(ids2)} cells")

# Now replicate _test_split steps and print
b_sub = np.array(["1"] * (len(ids1) + len(ids2)))
ids = np.concatenate([ids1, ids2])
true = data[np.ix_(var_idx, ids)]
labs = np.concatenate([np.ones(len(ids1), dtype=int), np.full(len(ids2), 2)])
gm = reduce_dimension(true, b_sub, 30)
obs_stat = ward_linkage_stat(gm[1], labs)
print(f"\nObserved Ward stat: {obs_stat:.4f}")
print(f"Observed PCs variance (trace of ESS): {((gm[1]-gm[1].mean(0))**2).sum():.1f}")

# On-genes
phi = poisson_dispersion_stats(true)
p_vals = 1 - norm.cdf(phi)
batch_rs = true.sum(axis=1)  # single batch
on_local = np.where((p_vals < 0.05) & (batch_rs > 0))[0]
print(f"\nOn-genes: {len(on_local)} of {len(var_idx)}")

# Fit model + 10 null draws
params = fit_model(true, on_local, b_sub, 30)
rng = np.random.default_rng(0)
null_stats = []
for i in range(10):
    s = _generate_null_statistic(true, params, on_local, b_sub, 30, gm, labs,
                                 posthoc=True, rng=rng)
    null_stats.append(s)
    print(f"  null {i}: {s:.4f}")
arr = np.asarray(null_stats)
print(f"\nNull stats: mean={arr.mean():.4f} sd={arr.std(ddof=1):.4f}")
print(f"Observed: {obs_stat:.4f}  → z = {(obs_stat - arr.mean())/arr.std(ddof=1):.3f}")
print(f"pval (1-Phi(z)): {1 - norm.cdf(obs_stat, loc=arr.mean(), scale=arr.std(ddof=1)):.4f}")

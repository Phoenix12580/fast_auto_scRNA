"""Dump Python's observed + 10 null Ward stats for the SAME root split,
matching exactly what debug_scshc_R_detailed.R does. Goal: pinpoint which
intermediate (reduce_dim, fit_model, on_genes, null-gen) diverges from R."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import norm

from scvalidate.scshc_py.core import (
    _deviance_feature_selection, _generate_null, _generate_null_statistic,
    fit_model, fit_model_batch, poisson_dispersion_stats, reduce_dimension,
    ward_linkage_stat, poisson_dev_batch,
)

BENCH = "F:/NMF_rewrite/scvalidate_rewrite/benchmark"

counts = mmread(f"{BENCH}/counts.mtx").tocsr()
genes = open(f"{BENCH}/genes.tsv").read().strip().split("\n")
cells = open(f"{BENCH}/cells.tsv").read().strip().split("\n")
cl_df = pd.read_csv(f"{BENCH}/scvalidate_clusters.csv", index_col=0).loc[cells]
leiden = cl_df["leiden"].astype(str).to_numpy()

data = np.asarray(counts.todense())
G, N = data.shape
print(f"data: {G}x{N}, leiden clusters: {sorted(set(leiden))}")

# var genes (top-2000 deviance)
dev = _deviance_feature_selection(data)
var_idx = np.argsort(-dev)[:2000]

# Root split same as R: {0,1,2,4,5,6} vs {3,7}
ids1 = np.where(np.isin(leiden, ["0", "1", "2", "4", "5", "6"]))[0]
ids2 = np.where(np.isin(leiden, ["3", "7"]))[0]
print(f"ids1: {len(ids1)}  ids2: {len(ids2)}")

ids = np.concatenate([ids1, ids2])
true = data[np.ix_(var_idx, ids)]
b_sub = np.array(["1"] * len(ids))
labs = np.concatenate([np.ones(len(ids1), dtype=int), np.full(len(ids2), 2)])

gm = reduce_dimension(true, b_sub, 30)
proj = gm[1]
print(f"Py reduce_dimension projection: {proj.shape}")
print(f"Py proj col means (first 5): {np.round(proj.mean(axis=0)[:5], 4)}")
print(f"Py proj abs range: [{abs(proj).min():.4f}, {abs(proj).max():.4f}]")
print(f"Py proj col-sd (first 5): {np.round(proj.std(axis=0, ddof=1)[:5], 4)}")

obs_stat = ward_linkage_stat(proj, labs)
print(f"Py observed Ward stat: {obs_stat:.4f}")

phi = poisson_dispersion_stats(true)
check_means = true.sum(axis=1)  # single batch
on_genes = np.where((1 - norm.cdf(phi) < 0.05) & (check_means != 0))[0]
print(f"Py on-genes: {len(on_genes)}")

params = fit_model(true, on_genes, b_sub, 30)
p1 = params["1"]
print(f"Py fit_model: lambdas head={np.round(p1.lambdas[:5], 4)}"
      f"  mus head={np.round(p1.mus[:5], 4)}")

# Diagnostics on the Gaussian-copula parameters
L = p1.on_cov_sqrt
print(f"Py on_cov_sqrt: {L.shape}  fro={np.linalg.norm(L, 'fro'):.2f}"
      f"  diag range=[{np.diag(L).min():.4f}, {np.diag(L).max():.4f}]")

cov_reconstructed = L @ L.T
diag_var = np.diag(cov_reconstructed)
print(f"Py implied on-gene variance: mean={diag_var.mean():.4f}"
      f"  max={diag_var.max():.4f}  min={diag_var.min():.4f}")

# R-style fit_model's diag(cov) is sigmas (log-normal scale) — report stats
print(f"Py mus: mean={p1.mus.mean():.4f}  range=[{p1.mus.min():.4f},{p1.mus.max():.4f}]")

# Null stats
rng = np.random.default_rng(0)
null_stats = []
for i in range(10):
    s = _generate_null_statistic(true, params, on_genes, b_sub, 30, gm, labs,
                                 posthoc=True, rng=rng)
    null_stats.append(s)
print(f"Py null stats (10 draws):")
print(" ", np.round(null_stats, 4))
arr = np.asarray(null_stats)
print(f"Py null mean={arr.mean():.4f}  sd={arr.std(ddof=1):.4f}")
print(f"Py z = (obs - mu)/sd = {(obs_stat - arr.mean())/arr.std(ddof=1):.4f}")
print(f"Py 1-Phi((obs - mu)/sd) = "
      f"{1 - norm.cdf(obs_stat, loc=arr.mean(), scale=arr.std(ddof=1)):.4f}")

# Peek at one null draw's on-gene rate scale vs real data magnitude
rng2 = np.random.default_rng(42)
null_counts, _ = _generate_null(true, params, on_genes, b_sub, rng2)
obs_on_col_sums = true[on_genes, :].sum(axis=0)
null_on_col_sums = null_counts[on_genes, :].sum(axis=0)
print(f"Real on-gene col_sum: mean={obs_on_col_sums.mean():.1f}"
      f"  median={np.median(obs_on_col_sums):.1f}")
print(f"Null on-gene col_sum: mean={null_on_col_sums.mean():.1f}"
      f"  median={np.median(null_on_col_sums):.1f}")

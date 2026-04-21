"""Compare Python vs R top-30 eigenvalues of rhos matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.linalg import eigh
from scipy.stats import norm

from scvalidate.scshc_py.core import (
    _deviance_feature_selection, poisson_dispersion_stats,
)

BENCH = "F:/NMF_rewrite/scvalidate_rewrite/benchmark"

counts = mmread(f"{BENCH}/counts.mtx").tocsr()
cells = open(f"{BENCH}/cells.tsv").read().strip().split("\n")
cl_df = pd.read_csv(f"{BENCH}/scvalidate_clusters.csv", index_col=0).loc[cells]
leiden = cl_df["leiden"].astype(str).to_numpy()

data = np.asarray(counts.todense())
dev = _deviance_feature_selection(data)
var_idx = np.argsort(-dev)[:2000]

ids1 = np.where(np.isin(leiden, ["0","1","2","4","5","6"]))[0]
ids2 = np.where(np.isin(leiden, ["3","7"]))[0]
ids = np.concatenate([ids1, ids2])
true = data[np.ix_(var_idx, ids)]

phi = poisson_dispersion_stats(true)
check_means = true.sum(axis=1)
on_genes = np.where((1 - norm.cdf(phi) < 0.05) & (check_means != 0))[0]
print(f"on_genes: {len(on_genes)}")

y = true
on_counts = y[on_genes, :].T
cov = np.cov(on_counts, rowvar=False, ddof=1)
means = on_counts.mean(axis=0)
with np.errstate(divide="ignore", invalid="ignore"):
    sigmas = np.log(((np.diag(cov) - means) / means ** 2) + 1.0)
    mus = np.log(means) - 0.5 * sigmas
mus_sum = mus[:, None] + mus[None, :]
sigmas_sum = sigmas[:, None] + sigmas[None, :]
with np.errstate(invalid="ignore", divide="ignore"):
    rhos = np.log(cov / np.exp(mus_sum + 0.5 * sigmas_sum) + 1.0)
rhos = np.where(np.isfinite(rhos), rhos, -10.0)
np.fill_diagonal(rhos, sigmas)

print(f"rhos: {rhos.shape}  symmetric? "
      f"{np.allclose(rhos, rhos.T)}")
print(f"rhos diag (sigmas) range: [{sigmas.min():.4f}, {sigmas.max():.4f}]")
print(f"rhos off-diag range: "
      f"[{rhos[np.triu_indices_from(rhos,1)].min():.4f}, "
      f"{rhos[np.triu_indices_from(rhos,1)].max():.4f}]")

vals_full, _ = eigh(rhos)
# ascending → get top 30 algebraic (desc)
top30_alg = vals_full[::-1][:30]
# top 30 by |λ| = top30 of |vals_full|, still returned desc by abs
idx_mag = np.argsort(np.abs(vals_full))[::-1][:30]
top30_mag = vals_full[idx_mag]

print(f"Top-30 algebraic (Python's current path): min={top30_alg.min():.4f}"
      f"  max={top30_alg.max():.4f}  num_pos={(top30_alg>0).sum()}")
print(f"  values: {np.round(top30_alg, 3)}")
print(f"Top-30 by |λ| (R's RSpectra path): "
      f"min={top30_mag.min():.4f}  max={top30_mag.max():.4f}"
      f"  num_pos={(top30_mag>0).sum()}")
print(f"  values: {np.round(top30_mag, 3)}")
print(f"Bottom-5 algebraic (most negative): "
      f"{np.round(vals_full[:5], 3)}")

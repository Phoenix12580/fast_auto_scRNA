"""M4 acceptance — Harmony 2.0 on 157k epithelia. Compares wall time
against the M4.2 baseline (94.1s Fixed, 100.6s Dynamic).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

import anndata as ad  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402

from scatlas import ext  # noqa: E402

EPITHELIA = Path("/mnt/f/NMF_rewrite/epithelia_full.h5ad")


def main() -> int:
    print("=" * 72)
    print("M4.3 BENCH — Harmony 2.0 on 157k epithelia")
    print("=" * 72)

    t0 = time.perf_counter()
    adata = ad.read_h5ad(EPITHELIA)
    print(f"[load] {adata.n_obs} cells × {adata.n_vars} genes in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((1e4 / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    print(f"[lognorm] {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    pca = TruncatedSVD(n_components=30, random_state=0).fit_transform(Xn).astype(np.float32)
    adata.obsm["X_pca"] = pca
    print(f"[pca] shape={pca.shape}, {time.perf_counter() - t0:.1f}s")

    adata.obs["batch"] = adata.obs["orig.ident"].astype(str)
    uniq, cnt = np.unique(adata.obs["batch"], return_counts=True)
    print(f"[batch] {dict(zip(uniq.tolist(), cnt.tolist()))}")

    # --- Fixed lambda ---
    print("\n--- Fixed lambda ---")
    t0 = time.perf_counter()
    out = ext.harmony(
        adata, batch_key="batch", use_rep="X_pca", n_clusters=100, seed=0
    )
    t_fixed = time.perf_counter() - t0
    print(f"[harmony fixed] {t_fixed:.1f}s, converged@iter={out['converged_at_iter']}")
    print(f"  iters of objective_harmony: {len(out['objective_harmony'])}")

    # --- Dynamic lambda ---
    print("\n--- Dynamic lambda ---")
    adata2 = adata.copy()
    t0 = time.perf_counter()
    out2 = ext.harmony(
        adata2, batch_key="batch", use_rep="X_pca",
        n_clusters=100, lambda_=None, alpha=0.2, seed=0,
    )
    t_dyn = time.perf_counter() - t0
    print(f"[harmony dynamic] {t_dyn:.1f}s, converged@iter={out2['converged_at_iter']}")

    print("\n" + "=" * 72)
    print(f"M4.2 baseline (pre-M4.3): Fixed 94.1s, Dynamic 100.6s")
    print(f"M4.3 now               : Fixed {t_fixed:.1f}s, Dynamic {t_dyn:.1f}s")
    print(f"  Fixed speedup   = {94.1 / t_fixed:.2f}x")
    print(f"  Dynamic speedup = {100.6 / t_dyn:.2f}x")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

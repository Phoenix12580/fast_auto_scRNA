"""M5.1 PCA benchmark — scatlas.pp.pca vs sklearn TruncatedSVD on 157k."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

import anndata as ad  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402

from scatlas import pp  # noqa: E402

EPITHELIA = Path("/mnt/f/NMF_rewrite/epithelia_full.h5ad")


def _principal_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    qu, _ = np.linalg.qr(u)
    qv, _ = np.linalg.qr(v)
    svals = np.linalg.svd(qu.T @ qv, compute_uv=False)
    svals = np.clip(svals, -1.0, 1.0)
    return float(np.degrees(np.arccos(svals.min())))


def main() -> int:
    print("=" * 72)
    print("M5.1 BENCH — randomized PCA on 157k epithelia × 16337 genes")
    print("=" * 72)

    t0 = time.perf_counter()
    adata = ad.read_h5ad(EPITHELIA)
    print(f"[load] {adata.n_obs} × {adata.n_vars} in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs[libs == 0] = 1
    Xn = sp.diags((1e4 / libs).astype(np.float32)) @ X.astype(np.float32)
    Xn.data = np.log1p(Xn.data)
    Xn = Xn.tocsr()
    print(f"[lognorm] {time.perf_counter() - t0:.1f}s, nnz={Xn.nnz}")

    # --- sklearn TruncatedSVD (scanpy's default path on sparse: n_iter=5)
    t0 = time.perf_counter()
    svd = TruncatedSVD(n_components=30, algorithm="randomized", n_iter=5, random_state=0)
    emb_sk = svd.fit_transform(Xn).astype(np.float32)
    t_sk = time.perf_counter() - t0
    print(f"\n[sklearn TruncatedSVD n_iter=5] {t_sk:.1f}s")

    # --- scatlas.pp.pca (defaults: n_power_iter=7)
    adata.X = Xn
    adata.obsm.pop("X_pca", None)
    t0 = time.perf_counter()
    out = pp.pca(adata, n_comps=30, random_state=0)
    t_sc = time.perf_counter() - t0
    emb_sc = out["embedding"]
    print(f"[scatlas.pp.pca defaults] {t_sc:.1f}s")

    # --- Subspace parity
    angle = _principal_angle_deg(emb_sc, emb_sk)
    print(f"\n[parity] max principal angle = {angle:.2f}°")

    # --- SV parity (relative)
    rel_sv = np.abs(out["singular_values"] - svd.singular_values_) / svd.singular_values_
    print(f"[parity] singular values max rel diff = {rel_sv.max():.3%}")

    print("\n" + "=" * 72)
    print(f"sklearn TruncatedSVD : {t_sk:.1f}s")
    print(f"scatlas.pp.pca       : {t_sc:.1f}s")
    print(f"  speedup            : {t_sk / t_sc:.2f}×")
    print(f"  subspace angle     : {angle:.2f}° (want < 5°)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

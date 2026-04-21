#!/usr/bin/env bash
# fast_auto_scRNA setup — one-shot installer for Linux / WSL2.
# Usage:
#   ./setup.sh              # fresh install
#   ./setup.sh --update     # re-build Rust kernels only (after git pull)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
UPDATE_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --update) UPDATE_ONLY=1 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

banner() {
    echo
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

check_cmd() {
    local cmd="$1" install_hint="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' not found."
        echo "  $install_hint"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 0. Prerequisite checks
# ---------------------------------------------------------------------------
banner "1/6  Checking prerequisites"

check_cmd cargo "install Rust via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
check_cmd rustc "install Rust (same as above)"

RUST_V=$(rustc --version | awk '{print $2}')
echo "  rustc $RUST_V"

if ! command -v uv >/dev/null 2>&1; then
    echo "  uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
fi
check_cmd uv "install uv (package manager)"
echo "  uv $(uv --version | awk '{print $2}')"

# ---------------------------------------------------------------------------
# 1. venv
# ---------------------------------------------------------------------------
if [ "$UPDATE_ONLY" -eq 0 ]; then
    banner "2/6  Creating Python venv ($VENV_DIR)"
    if [ -d "$VENV_DIR" ]; then
        echo "  venv exists, re-using"
    else
        uv venv --python 3.10 "$VENV_DIR"
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  python: $(which python) ($(python --version))"

# ---------------------------------------------------------------------------
# 2. Python deps
# ---------------------------------------------------------------------------
if [ "$UPDATE_ONLY" -eq 0 ]; then
    banner "3/6  Installing Python dependencies"
    uv pip install \
        "maturin>=1.7" \
        "numpy>=1.26" \
        "scipy>=1.11" \
        "pandas>=2.0" \
        "anndata>=0.10" \
        "scanpy>=1.11" \
        "scikit-learn>=1.3" \
        "matplotlib>=3.8" \
        "leidenalg" \
        "python-igraph" \
        "umap-learn" \
        "pytest>=8" \
        "rdata"
fi

# ---------------------------------------------------------------------------
# 3. Build scatlas (Rust kernels #1)
# ---------------------------------------------------------------------------
banner "4/6  Building scatlas (Rust kernels)"
cd "$REPO_ROOT/scatlas/crates/scatlas-py"
maturin develop --release
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 4. Build scvalidate_rust (Rust kernels #2)
# ---------------------------------------------------------------------------
banner "5/6  Building scvalidate_rust (Rust kernels)"
cd "$REPO_ROOT/scvalidate_rewrite/scvalidate_rust"
maturin develop --release
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 5. Install Python editable packages
# ---------------------------------------------------------------------------
if [ "$UPDATE_ONLY" -eq 0 ]; then
    banner "6/6  Installing Python packages (editable)"
    uv pip install -e "$REPO_ROOT/scvalidate_rewrite"
    # scatlas_pipeline doesn't have a pyproject.toml (yet) — just add to PYTHONPATH via a .pth
    SITE_PKGS=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    echo "$REPO_ROOT" > "$SITE_PKGS/fast_auto_scrna.pth"
    echo "  wrote $SITE_PKGS/fast_auto_scrna.pth → $REPO_ROOT"
fi

# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------
banner "Smoke test"
python - <<'PY'
import numpy as np
import scipy.sparse as sp
import anndata as ad
import tempfile, os

print("  importing scatlas...", end=" ", flush=True)
from scatlas import pp, ext, tl, metrics  # noqa: F401
print("OK")

print("  importing scvalidate...", end=" ", flush=True)
from scvalidate.recall_py import find_clusters_recall  # noqa: F401
print("OK")

print("  importing scatlas_pipeline...", end=" ", flush=True)
from scatlas_pipeline import run_pipeline  # noqa: F401
print("OK")

print("  running mini pipeline on synthetic data...")
rng = np.random.default_rng(0)
X = sp.csr_matrix(rng.poisson(1, (500, 300)).astype(np.float32))
a = ad.AnnData(X=X)
a.obs["batch"] = np.where(np.arange(500) < 250, "A", "B")
with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
    tmp = f.name
a.write_h5ad(tmp)
try:
    out = run_pipeline(
        tmp, batch_key="batch",
        min_genes=5, min_cells=1, max_pct_mt=100,
        pca_n_comps=10, run_harmony=True,
        run_umap=True, run_leiden=True,
        leiden_target_n=(2, 10),
        run_recall=False, run_metrics=False,
    )
    print(f"    pipeline ok — {out.n_obs} cells, "
          f"{out.obs['leiden'].nunique()} clusters, "
          f"X_umap {out.obsm['X_umap'].shape}")
finally:
    os.unlink(tmp)
PY

banner "DONE"
echo "Activate env:  source .venv/bin/activate"
echo "Quickstart:    python scatlas_pipeline/benchmarks/compare_pancreas_sub.py"

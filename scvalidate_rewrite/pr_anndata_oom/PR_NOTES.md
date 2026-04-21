# PR prep: anndata-oom Windows support

Target repo: https://github.com/omicverse/anndata-oom

## Problem observed (2026-04-19)

On a clean Windows 10 + Python 3.11 env:

```
pip install anndataoom
```

fails because:

1. **No Windows wheels on PyPI.** `https://pypi.org/pypi/anndataoom/json` lists only Linux + macOS wheels; Windows falls back to sdist and must compile from source.
2. **Source build chains into HDF5 + Rust.** The Rust `hdf5-metno-src` crate tries to compile HDF5 via CMake; CMake's try-compile step invokes MSVC's `-Brepro` flag, which fails on BuildTools 14.44 + CMake 4.3.1 with "Detecting C compiler ABI info - failed".
3. **Default VS generator fails.** Forcing `CMAKE_GENERATOR=Ninja` bypasses the `-Brepro` try-compile issue, but still needs the full MSVC+Rust+cmake+ninja+maturin toolchain activated via `vcvarsall x64` — not documented in the README.
4. **Current README** claims "works on Windows" — misleading given the above.

## Proposed PR changes

### 1. `README.md` — add a Windows install section

See `README_windows_patch.md` in this dir for the proposed diff.

### 2. `.github/workflows/wheels.yml` — add Windows matrix to cibuildwheel

See `wheels.yml` in this dir.

Windows wheels via cibuildwheel resolve the entire story: users never see the source build path.

### 3. Optional: fall-back path for users who do need to build from source

Add a `scripts/install_windows.bat` (or document it in README) that activates
MSVC, installs cmake+ninja, sets `CMAKE_GENERATOR=Ninja`, then runs
`pip install --no-build-isolation anndataoom`. Draft at
`install_windows.bat`.

## TODO before filing PR

- [ ] Fork omicverse/anndata-oom to user's account
- [ ] Clone and apply these changes on a feature branch
- [ ] Run the cibuildwheel workflow locally / in the fork to confirm it produces
      a Windows wheel
- [ ] Test the produced wheel on a clean Windows VM (`pip install <wheel>` +
      `python -c "import anndataoom as oom; oom.read(...)"`)
- [ ] Open PR with description linking the Windows-install reproduction above

## Upstream issue to reference

No existing issue confirms the Windows failure; first comment on PR should link
a minimal reproduction (`install_aoom_win.bat` in `benchmark/`).

# Proposed README patch — add Windows install section

Insert **after** the existing "Installation" section:

---

### Windows

Pre-built wheels are published for Windows (cibuildwheel matrix in CI). Install
the same way as Linux/macOS:

```
pip install anndataoom
```

If pip falls back to building from source (e.g. no matching wheel for your
Python version), you need the full MSVC + Rust + HDF5 toolchain. The cleanest
sequence:

```bat
rem 1. Install Rust (once): https://rustup.rs
rem 2. Install MSVC Build Tools 2022 (C++ workload) and activate:
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

rem 3. Force Ninja generator — the default VS generator fails CABI detection
rem    with BuildTools 14.44 + CMake 4.3.1 + `-Brepro` try-compile.
set CMAKE_GENERATOR=Ninja

rem 4. Install build deps into the active env
pip install --upgrade cmake ninja maturin setuptools wheel

rem 5. Install anndataoom without PEP 517 build isolation so the build
rem    inherits the cmake/ninja/maturin from step 4.
pip install --no-build-isolation anndataoom
```

Expect ~10–20 minutes for the HDF5 source compile. A one-shot script is
provided in `scripts/install_windows.bat`.

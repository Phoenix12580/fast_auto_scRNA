@echo off
rem One-shot Windows install for anndataoom (source build path).
rem Requires MSVC BuildTools 2022 and Rust already installed.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo ERROR: vcvarsall.bat failed. Install MSVC BuildTools 2022 first.
    exit /b 1
)

where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not on PATH after vcvarsall.
    exit /b 1
)

pip install --upgrade pip setuptools wheel maturin cmake ninja
if errorlevel 1 exit /b 1

set CMAKE_GENERATOR=Ninja

pip install --no-build-isolation anndataoom

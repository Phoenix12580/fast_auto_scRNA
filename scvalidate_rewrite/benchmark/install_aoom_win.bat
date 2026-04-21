@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo vcvarsall.bat FAILED
    exit /b 1
)
echo === MSVC env activated ===
where cl
echo.
echo === Ensure build deps exist in active env (PEP 517 isolation would skip them) ===
pip install --upgrade pip setuptools wheel maturin cmake ninja

rem Force Ninja generator. The Rust cmake-rs crate defaults to the VS
rem generator on Windows, which fails CABI detection with BuildTools 14.44
rem (-Brepro flag tripping the try-compile step). Ninja bypasses that.
set CMAKE_GENERATOR=Ninja

echo.
echo === Starting pip install anndataoom (no build isolation, Ninja gen) ===
pip install --no-build-isolation anndataoom

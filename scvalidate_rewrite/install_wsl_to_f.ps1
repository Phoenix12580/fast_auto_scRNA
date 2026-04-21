# Install WSL2 + Ubuntu 22.04 to F:\WSL (instead of C:)
# USAGE: open PowerShell as Administrator, then:
#   powershell -ExecutionPolicy Bypass -File F:\scvalidate_rewrite\install_wsl_to_f.ps1

$ErrorActionPreference = "Stop"
$WSLRoot = "F:\WSL"
$DistroName = "Ubuntu"
$TarPath = "$WSLRoot\ubuntu-jammy-rootfs.tar.gz"
$TarUrl = "https://cloud-images.ubuntu.com/wsl/jammy/current/ubuntu-jammy-wsl-amd64-wsl.rootfs.tar.gz"

Write-Host "=== Step 1/5: Install WSL base (may require reboot) ===" -ForegroundColor Cyan
wsl --install --no-distribution
if ($LASTEXITCODE -ne 0) {
    Write-Host "wsl --install exit=$LASTEXITCODE. If it asks for reboot, reboot and re-run." -ForegroundColor Yellow
}

Write-Host "=== Step 2/5: Set default to WSL2 ===" -ForegroundColor Cyan
wsl --set-default-version 2

Write-Host "=== Step 3/5: Create F:\WSL and download Ubuntu 22.04 rootfs (~700 MB) ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $WSLRoot -Force | Out-Null
if (-not (Test-Path $TarPath)) {
    Write-Host "Downloading $TarUrl ..."
    curl.exe -L --fail -o $TarPath $TarUrl
} else {
    Write-Host "rootfs tar already present at $TarPath, skipping download."
}

Write-Host "=== Step 4/5: Import Ubuntu to F:\WSL\Ubuntu ===" -ForegroundColor Cyan
wsl --import $DistroName "$WSLRoot\$DistroName" $TarPath --version 2

Write-Host "=== Step 5/5: Verify ===" -ForegroundColor Cyan
wsl --list --verbose
Write-Host ""
Write-Host "Ubuntu installed to $WSLRoot\$DistroName" -ForegroundColor Green
Write-Host "Launch with:    wsl -d $DistroName" -ForegroundColor Green
Write-Host ""
Write-Host "After launching, create a non-root user (one-time, optional):" -ForegroundColor Yellow
Write-Host "  adduser <username>" -ForegroundColor White
Write-Host "  usermod -aG sudo <username>" -ForegroundColor White
Write-Host "  printf '[user]\ndefault=<username>\n' >> /etc/wsl.conf" -ForegroundColor White
Write-Host "  exit  (then from Windows)  wsl --terminate $DistroName" -ForegroundColor White

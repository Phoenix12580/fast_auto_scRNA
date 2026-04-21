# Install WSL2 Ubuntu — install to default C:, then relocate to F:\WSL\Ubuntu
# USAGE: admin PowerShell, then:
#   & F:\scvalidate_rewrite\install_wsl_to_f_v2.ps1

$ErrorActionPreference = "Stop"
$WSLRoot = "F:\WSL"
$DistroName = "Ubuntu"
$ExportTar = "$WSLRoot\Ubuntu-export.tar"

Write-Host "=== Step 1/6: Install WSL2 Ubuntu to default C: location ===" -ForegroundColor Cyan
Write-Host "A Microsoft Store / Canonical download will start; follow any prompts." -ForegroundColor Yellow
wsl --install -d Ubuntu

Write-Host ""
Write-Host "=== MANUAL STEP ===" -ForegroundColor Magenta
Write-Host "When the Ubuntu window pops up, it will ask you to create a UNIX username + password." -ForegroundColor Magenta
Write-Host "Do that once, then type 'exit' to leave Ubuntu." -ForegroundColor Magenta
Write-Host "After you've returned to this PowerShell prompt, press Enter to continue." -ForegroundColor Magenta
Read-Host "Press Enter when Ubuntu initial setup is done"

Write-Host "=== Step 2/6: Shutdown WSL ===" -ForegroundColor Cyan
wsl --shutdown

Write-Host "=== Step 3/6: Create F:\WSL and export current Ubuntu ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $WSLRoot -Force | Out-Null
wsl --export Ubuntu $ExportTar

Write-Host "=== Step 4/6: Unregister C-drive Ubuntu ===" -ForegroundColor Cyan
wsl --unregister Ubuntu

Write-Host "=== Step 5/6: Import Ubuntu to F:\WSL\Ubuntu ===" -ForegroundColor Cyan
wsl --import $DistroName "$WSLRoot\$DistroName" $ExportTar --version 2

Write-Host "=== Step 6/6: Cleanup + verify ===" -ForegroundColor Cyan
Remove-Item $ExportTar
wsl --set-default $DistroName
wsl --list --verbose
Write-Host ""
Write-Host "Done. Ubuntu now lives at $WSLRoot\$DistroName\ext4.vhdx" -ForegroundColor Green
Write-Host "Launch: wsl -d $DistroName   (or just:  wsl)" -ForegroundColor Green
Write-Host ""
Write-Host "Note: after --import, root is the default user. To restore your " -ForegroundColor Yellow
Write-Host "username as default, in WSL run:" -ForegroundColor Yellow
Write-Host "  printf '[user]\ndefault=<your-username>\n' | sudo tee -a /etc/wsl.conf" -ForegroundColor White
Write-Host "  exit    # then from Windows:  wsl --terminate Ubuntu ; wsl" -ForegroundColor White

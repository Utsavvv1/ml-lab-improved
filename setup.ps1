#!/usr/bin/env pwsh
Param(
    [switch]$SkipInstall
#!/usr/bin/env pwsh
Param(
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
Write-Host "== ml-lab-improved: setup.ps1 =="

Write-Host "Creating virtual environment .venv..."
python -m venv .venv

Write-Host "Upgrading pip in .venv..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

if (-not $SkipInstall) {
    if (Test-Path "requirements.txt") {
        Write-Host "Installing dependencies from requirements.txt..."
        & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    }
    else {
        Write-Host "No requirements.txt found — skipping pip install."
    }
}

Write-Host "`nDone. Next steps:"
Write-Host ' - To activate the venv in this PowerShell session run: .\.venv\Scripts\Activate'
Write-Host ' - In VS Code, open Command Palette (Ctrl+Shift+P) → "Python: Select Interpreter" → choose the interpreter at ".\.venv\Scripts\python.exe".'
Write-Host ' - If you want to skip installing packages, run: powershell -ExecutionPolicy Bypass -File .\setup.ps1 -SkipInstall'
)

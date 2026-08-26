# One-shot environment setup for Windows PowerShell.
#
#   .\setup.ps1
#
# Creates a virtual environment, installs dependencies, and verifies the data
# foundation. Safe to re-run: an existing .venv is reused, not rebuilt.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "=== Predictive Maintenance : setup ===" -ForegroundColor Cyan
Write-Host ""

# --- Python check ------------------------------------------------------------
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python not found on PATH. Install Python 3.10 or newer first." -ForegroundColor Red
    exit 1
}
$version = (python -c "import sys; print('%d.%d' % sys.version_info[:2])")
Write-Host "Python $version detected" -ForegroundColor Green

# --- Virtual environment -----------------------------------------------------
if (Test-Path ".venv") {
    Write-Host "Reusing existing .venv"
} else {
    Write-Host "Creating .venv ..."
    python -m venv .venv
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# --- Dependencies ------------------------------------------------------------
Write-Host "Upgrading pip ..."
& $venvPython -m pip install --upgrade pip --quiet

Write-Host "Installing requirements (this takes a few minutes, TensorFlow is large) ..."
& $venvPython -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dependency install failed." -ForegroundColor Red
    exit 1
}
Write-Host "Dependencies installed" -ForegroundColor Green

# --- Verify ------------------------------------------------------------------
Write-Host ""
Write-Host "Verifying the data foundation ..."
& $venvPython verify_foundation.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Verification failed. The dataset may be missing or corrupt." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Activate the environment:"
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then train the models:"
Write-Host "    python -m src.train_classifier" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then launch the dashboard:"
Write-Host "    streamlit run dashboard/app.py" -ForegroundColor Yellow
Write-Host ""

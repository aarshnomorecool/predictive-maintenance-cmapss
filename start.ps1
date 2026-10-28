# Launch the Predictive Maintenance Dashboard in one go.
# Usage:
#   .\start.ps1
# or:
#   python -m api.server

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Turbofan Health Monitor & Bob the Mechanic Server      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

$python = "python"
if (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe") {
    $python = "$PSScriptRoot\.venv\Scripts\python.exe"
    Write-Host "Using virtual environment: .venv" -ForegroundColor Green
} else {
    Write-Host "Using system Python on PATH" -ForegroundColor Yellow
}

$port = 8050
$url = "http://127.0.0.1:$port"

Write-Host "Starting dashboard server at $url ..." -ForegroundColor Green
Write-Host "Press Ctrl+C at any time to stop the server." -ForegroundColor Gray
Write-Host ""

# Launch browser after a brief delay in the background
Start-Job -ScriptBlock {
    param($u)
    Start-Sleep -Seconds 2
    Start-Process $u
} -ArgumentList $url | Out-Null

# Run server in the foreground
& $python -m api.server

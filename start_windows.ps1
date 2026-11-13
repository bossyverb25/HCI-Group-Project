# ==============================================================================
# The Invisible Interpreter — Windows Quick Start Script (PowerShell)
# ==============================================================================
# Run in PowerShell:
#   .\start_windows.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Error ".venv not found. Please run .\setup_windows.ps1 (or setup_windows.bat) first."
}

# Ensure Ollama server is running in the background
$OllamaRunning = $false
try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
    if ($resp.StatusCode -eq 200) { $OllamaRunning = $true }
} catch {}

if (-not $OllamaRunning) {
    $OllamaExe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $OllamaExe -and (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
        $OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    }
    if ($OllamaExe) {
        Write-Host "Starting Ollama background server..." -ForegroundColor Cyan
        Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 1
    }
}

Write-Host "Starting The Invisible Interpreter on http://localhost:8000 ..." -ForegroundColor Green
& ".venv\Scripts\python.exe" app.py

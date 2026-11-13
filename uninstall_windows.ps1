# ==============================================================================
# The Invisible Interpreter — Windows Uninstaller & Cleanup Script (PowerShell)
# ==============================================================================
# Removes:
#   1. Downloaded Ollama models (llama3.2:1b, llama3.2:3b) and ~/.ollama cache
#   2. Cached Faster-Whisper model weights (~/.cache/huggingface/hub/models--Systran--faster-whisper*)
#   3. Local Python virtual environment (.venv)
#   4. (Optional prompt) Uninstalls the Ollama desktop application via winget
# ==============================================================================

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

Write-Host "=== [1/4] Removing downloaded Ollama models (llama3.2:3b, llama3.2:1b) ===" -ForegroundColor Cyan
$OllamaExe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $OllamaExe -and (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
    $OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
}

if ($OllamaExe) {
    & $OllamaExe rm llama3.2:3b 2>$null
    & $OllamaExe rm llama3.2:1b 2>$null
}

Write-Host ""
Write-Host "=== [2/4] Removing Faster-Whisper cached model weights ===" -ForegroundColor Cyan
$HfCache = "$env:USERPROFILE\.cache\huggingface\hub"
if (Test-Path $HfCache) {
    Get-ChildItem -Path $HfCache -Filter "models--Systran--faster-whisper*" -Directory | ForEach-Object {
        Write-Host "Deleting $($_.FullName)..."
        Remove-Item -Recurse -Force $_.FullName
    }
}

Write-Host ""
Write-Host "=== [3/4] Removing Python virtual environment (.venv) ===" -ForegroundColor Cyan
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force ".venv"
    Write-Host "Removed .venv" -ForegroundColor Green
}

Write-Host ""
$UninstallApp = Read-Host "Do you also want to completely uninstall the Ollama application and delete ~/.ollama? (y/N)"
if ($UninstallApp -match "^[Yy]$") {
    Write-Host "=== [4/4] Uninstalling Ollama application ===" -ForegroundColor Cyan
    Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    winget uninstall --id Ollama.Ollama
    if (Test-Path "$env:USERPROFILE\.ollama") {
        Remove-Item -Recurse -Force "$env:USERPROFILE\.ollama"
        Write-Host "Deleted $env:USERPROFILE\.ollama" -ForegroundColor Green
    }
} else {
    Write-Host "Kept Ollama app installed (only removed project models and .venv)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Cleanup complete!" -ForegroundColor Green

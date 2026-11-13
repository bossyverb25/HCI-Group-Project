# ==============================================================================
# The Invisible Interpreter (Prototype v1) — Windows Setup & Runner (PowerShell)
# ==============================================================================
# Run this script in PowerShell from inside the cloned GitHub repository:
#   .\setup_windows.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$OllamaModel = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "llama3.2:3b" }

Write-Host "=== [1/4] Checking Python 3.12 (required for faster-whisper wheels on Windows) ===" -ForegroundColor Cyan

function Get-CompatiblePython {
    # Check if py launcher has 3.12, 3.11, or 3.13
    foreach ($ver in @("-3.12", "-3.11", "-3.13")) {
        try {
            $out = & py $ver -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                return $out.Trim()
            }
        } catch {}
    }
    # Check default python if it is 3.10..3.13
    try {
        $verStr = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $verStr -in @("3.10", "3.11", "3.12", "3.13")) {
            $exe = & python -c "import sys; print(sys.executable)" 2>$null
            return $exe.Trim()
        }
    } catch {}
    return $null
}

$PythonExe = Get-CompatiblePython
if (-not $PythonExe) {
    Write-Host "Compatible Python (3.11-3.13) not found. Installing Python 3.12 via winget..." -ForegroundColor Yellow
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    # Refresh PATH in current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    $PythonExe = Get-CompatiblePython
    if (-not $PythonExe) {
        $FallbackPath = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
        if (Test-Path $FallbackPath) {
            $PythonExe = $FallbackPath
        } else {
            Write-Error "Python 3.12 installation finished, but python.exe wasn't found in PATH yet. Please close and reopen PowerShell and re-run .\setup_windows.ps1"
        }
    }
}

Write-Host "Using Python: $PythonExe" -ForegroundColor Green

# Recreate .venv if it was built with Python 3.14 or doesn't exist
if (Test-Path ".venv\Scripts\python.exe") {
    $venvVer = & ".venv\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($venvVer -eq "3.14") {
        Write-Host "Removing existing Python 3.14 .venv and rebuilding with $PythonExe..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force .venv
    }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating Python virtual environment (.venv)..."
    & $PythonExe -m venv .venv
}

Write-Host "Installing Python dependencies from requirements.txt..."
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "=== [2/4] Checking Ollama installation ===" -ForegroundColor Cyan
$OllamaExe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $OllamaExe) {
    $DefaultOllama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path $DefaultOllama) {
        $OllamaExe = $DefaultOllama
    }
}

if (-not $OllamaExe) {
    Write-Host "Ollama not found. Installing Ollama via winget..." -ForegroundColor Yellow
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    $DefaultOllama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path $DefaultOllama) {
        $OllamaExe = $DefaultOllama
    } else {
        $OllamaExe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    }
}

Write-Host "Using Ollama: $OllamaExe" -ForegroundColor Green

Write-Host ""
Write-Host "=== [3/4] Starting Ollama service & pulling model ($OllamaModel) ===" -ForegroundColor Cyan
$OllamaRunning = $false
try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if ($resp.StatusCode -eq 200) { $OllamaRunning = $true }
} catch {}

if (-not $OllamaRunning) {
    Write-Host "Starting Ollama background server..."
    Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) { break }
        } catch {}
    }
}

Write-Host "Ensuring model '$OllamaModel' is downloaded..."
& $OllamaExe pull $OllamaModel

Write-Host ""
Write-Host "=== [4/4] Launching The Invisible Interpreter on http://localhost:8000 ===" -ForegroundColor Cyan
& ".venv\Scripts\python.exe" app.py

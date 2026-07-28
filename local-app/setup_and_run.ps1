# ============================================================
# KEIKO Local Lab - PowerShell Setup & Run Script
# ============================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$PythonExe = $null

if ($env:KEIKO_VENV -and (Test-Path "$env:KEIKO_VENV\Scripts\python.exe")) {
    $PythonExe = "$env:KEIKO_VENV\Scripts\python.exe"
    Write-Host "Using KEIKO_VENV environment variable: $PythonExe" -ForegroundColor Green
} elseif (Test-Path "P:\Dependencies\keiko_venv\Scripts\python.exe") {
    $PythonExe = "P:\Dependencies\keiko_venv\Scripts\python.exe"
    Write-Host "Found workspace venv at P:\Dependencies\keiko_venv" -ForegroundColor Green
} elseif ($env:VIRTUAL_ENV -and (Test-Path "$env:VIRTUAL_ENV\Scripts\python.exe")) {
    $PythonExe = "$env:VIRTUAL_ENV\Scripts\python.exe"
    Write-Host "Using active virtual environment: $PythonExe" -ForegroundColor Green
} elseif (Test-Path "$ScriptDir\venv\Scripts\python.exe") {
    $PythonExe = "$ScriptDir\venv\Scripts\python.exe"
    Write-Host "Found local venv." -ForegroundColor Green
} elseif (Test-Path "$ScriptDir\..\venv\Scripts\python.exe") {
    $PythonExe = "$ScriptDir\..\venv\Scripts\python.exe"
    Write-Host "Found parent venv." -ForegroundColor Green
} else {
    $sysPy = Get-Command python -ErrorAction SilentlyContinue
    if ($sysPy) {
        $PythonExe = "python"
        Write-Host "Using system Python." -ForegroundColor Yellow
    } else {
        Write-Host "ERROR: No Python installation or virtual environment found." -ForegroundColor Red
        Exit 1
    }
}

$env:PYTHONPATH = "$ScriptDir;$ScriptDir\.."
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"

& $PythonExe "$ScriptDir\run.py" $args

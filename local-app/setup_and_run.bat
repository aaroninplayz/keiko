@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: KEIKO Local Lab - Setup & Run Script (Windows Batch)
:: ============================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE="

if defined KEIKO_VENV if exist "%KEIKO_VENV%\Scripts\python.exe" (
    set "PYTHON_EXE=%KEIKO_VENV%\Scripts\python.exe"
    goto :found_python
)

if exist "P:\Dependencies\keiko_venv\Scripts\python.exe" (
    set "PYTHON_EXE=P:\Dependencies\keiko_venv\Scripts\python.exe"
    goto :found_python
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
    goto :found_python
)

if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%venv\Scripts\python.exe"
    goto :found_python
)

if exist "%SCRIPT_DIR%..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%..\venv\Scripts\python.exe"
    goto :found_python
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
    goto :found_python
)

echo ERROR: No Python installation or virtual environment found.
pause
exit /b 1

:found_python
echo Using Python: !PYTHON_EXE!

set "PYTHONPATH=%SCRIPT_DIR%;%SCRIPT_DIR%.."
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"

if "%~1"=="" goto :start_server
if "%~1"=="--restart" goto :start_server

:menu
echo.
echo ============================================================
echo               KEIKO LOCAL LAB CONTROL MENU
echo ============================================================
echo  [1] Launch Server
echo  [2] Restart Server
echo  [3] Run Diagnostics Check
echo  [4] Exit
echo ============================================================
set /p CHOICE="Select an option (1-4): "

if "%CHOICE%"=="1" goto :start_server
if "%CHOICE%"=="2" goto :restart_server
if "%CHOICE%"=="3" goto :run_check
if "%CHOICE%"=="4" exit /b 0

echo Invalid choice.
goto :menu

:run_check
"!PYTHON_EXE!" "%SCRIPT_DIR%run.py" --check
goto :menu

:restart_server
echo Stopping existing KEIKO python process...
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run.py*' } | Stop-Process -Force" >nul 2>&1
taskkill /F /FI "COMMANDLINE eq *run.py*" >nul 2>&1
timeout /t 1 /nobreak >nul
goto :start_server

:start_server
echo Launching KEIKO server...
"!PYTHON_EXE!" "%SCRIPT_DIR%run.py" %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo Keiko launcher exited with status code %ERRORLEVEL%.
)
echo.
echo Server stopped or restarted.
set /p RESTART_CHOICE="Restart server now? (y/n): "
if /i "%RESTART_CHOICE%"=="y" goto :start_server
pause

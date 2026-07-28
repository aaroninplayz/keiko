@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: KEIKO TALENT INTELLIGENCE PLATFORM - INSTANT STARTUP SCRIPT
:: ============================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ============================================================
echo           KEIKO TALENT INTELLIGENCE PLATFORM              
echo ============================================================

set "PYTHON_EXE="

if exist "P:\Dependencies\keiko_venv\Scripts\python.exe" (
    set "PYTHON_EXE=P:\Dependencies\keiko_venv\Scripts\python.exe"
    goto :found_python
)

if exist "%SCRIPT_DIR%local-app\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%local-app\venv\Scripts\python.exe"
    goto :found_python
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
    goto :found_python
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
    goto :found_python
)

echo [!] ERROR: No Python virtual environment found.
pause
exit /b 1

:found_python
echo [+] Python Environment: !PYTHON_EXE!
echo [+] Launching Keiko Local Server...
echo [+] Server Dashboard URL: http://localhost:8000/static/dashboard.html
echo ============================================================
echo.

:: Automatically launch default web browser after 2 seconds
start "" "http://localhost:8000/static/dashboard.html"

:: Execute FastAPI server launcher
"!PYTHON_EXE!" "%SCRIPT_DIR%local-app\run.py"

pause

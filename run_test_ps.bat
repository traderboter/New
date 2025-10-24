@echo off
REM ===================================================
REM Pattern Test Runner for Windows PowerShell
REM ===================================================

echo ===================================================
echo Pattern Test - PowerShell Version
echo ===================================================
echo.

if "%~1"=="" (
    echo Usage: run_test_ps.bat doji
    pause
    exit /b 1
)

REM Get script directory and go there
cd /d "%~dp0"

REM Activate venv
call venv\Scripts\activate.bat

REM Run with full path
python "%~dp0signal_generation\tests\test_pattern.py" --pattern "%~1" --data-dir "%~dp0historical\BTC-USDT"

pause

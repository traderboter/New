@echo off
REM ===================================================
REM Pattern Test Runner for Windows
REM ===================================================
REM
REM Usage:
REM   run_test_pattern.bat doji
REM   run_test_pattern.bat hammer
REM   run_test_pattern.bat "morning star"
REM

echo ===================================================
echo Pattern Test Runner
echo ===================================================
echo.

REM Check if pattern name is provided
if "%~1"=="" (
    echo Error: Please provide pattern name
    echo.
    echo Usage:
    echo   run_test_pattern.bat doji
    echo   run_test_pattern.bat hammer
    echo   run_test_pattern.bat "morning star"
    echo.
    echo Available patterns:
    echo   Candlestick: doji, hammer, engulfing, shooting star, etc.
    echo   Chart: double top bottom, triangle, wedge, head shoulders
    echo.
    pause
    exit /b 1
)

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo Current directory: %CD%
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found at venv\Scripts\
    echo Please create venv first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install pandas numpy scipy TA-Lib
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if data directory exists
if not exist "historical\BTC-USDT" (
    echo Warning: Data directory not found: historical\BTC-USDT
    echo.
    echo Please ensure your data files are in:
    echo   historical\BTC-USDT\5m.csv
    echo   historical\BTC-USDT\15m.csv
    echo   historical\BTC-USDT\1h.csv
    echo   historical\BTC-USDT\4h.csv
    echo.
    pause
)

REM Set PYTHONPATH
set PYTHONPATH=%CD%

REM Run the test
echo.
echo ===================================================
echo Testing pattern: %~1
echo ===================================================
echo.

python signal_generation\tests\test_pattern.py --pattern "%~1" --data-dir historical\BTC-USDT

echo.
echo ===================================================
echo Test completed
echo ===================================================
echo.

REM Pause so user can see the results
pause

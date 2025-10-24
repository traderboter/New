# ===================================================
# Pattern Test Runner for PowerShell
# ===================================================
# Usage:
#   .\run_test.ps1 doji
#   .\run_test.ps1 "morning star"
# ===================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$PatternName,

    [Parameter(Mandatory=$false)]
    [string]$DataDir = "historical\BTC-USDT"
)

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Pattern Test Runner - PowerShell" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Current directory: $PWD" -ForegroundColor Gray
Write-Host "Testing pattern: $PatternName" -ForegroundColor Yellow
Write-Host ""

# Check if venv exists
if (-not (Test-Path "venv\Scripts\activate.ps1")) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create venv first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv" -ForegroundColor Gray
    Write-Host "  .\venv\Scripts\activate.ps1" -ForegroundColor Gray
    Write-Host "  pip install pandas numpy scipy TA-Lib" -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Gray
& "venv\Scripts\activate.ps1"

# Check if data directory exists
$DataPath = Join-Path $ScriptDir $DataDir
if (-not (Test-Path $DataPath)) {
    Write-Host "Warning: Data directory not found: $DataPath" -ForegroundColor Yellow
    Write-Host ""
}

# Run the test
Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Running test..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

$TestScript = Join-Path $ScriptDir "signal_generation\tests\test_pattern.py"
python $TestScript --pattern $PatternName --data-dir $DataDir

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Test completed" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"

# ===================================================
# Rename CSV files to match expected format
# ===================================================
# This script renames:
#   5min.csv  -> 5m.csv
#   15min.csv -> 15m.csv
#   1hour.csv -> 1h.csv
#   4hour.csv -> 4h.csv
# ===================================================

param(
    [Parameter(Mandatory=$false)]
    [string]$DataDir = "historical\BTC-USDT"
)

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "CSV File Renamer for Pattern Testing" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Check if directory exists
if (-not (Test-Path $DataDir)) {
    Write-Host "Error: Directory not found: $DataDir" -ForegroundColor Red
    exit 1
}

Write-Host "Data directory: $DataDir" -ForegroundColor Yellow
Write-Host ""

# Define rename mappings
$renameMappings = @{
    "5min.csv"  = "5m.csv"
    "15min.csv" = "15m.csv"
    "1hour.csv" = "1h.csv"
    "4hour.csv" = "4h.csv"
}

Write-Host "Checking files to rename..." -ForegroundColor Cyan
Write-Host ""

$renamed = 0
$skipped = 0

foreach ($oldName in $renameMappings.Keys) {
    $newName = $renameMappings[$oldName]
    $oldPath = Join-Path $DataDir $oldName
    $newPath = Join-Path $DataDir $newName

    if (Test-Path $oldPath) {
        if (Test-Path $newPath) {
            Write-Host "  ⚠️  Skipping: $newName already exists" -ForegroundColor Yellow
            $skipped++
        } else {
            try {
                Rename-Item -Path $oldPath -NewName $newName
                Write-Host "  ✓ Renamed: $oldName -> $newName" -ForegroundColor Green
                $renamed++
            } catch {
                Write-Host "  ✗ Failed to rename: $oldName" -ForegroundColor Red
                Write-Host "    Error: $_" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "  - Not found: $oldName" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Renamed: $renamed files" -ForegroundColor Green
Write-Host "  Skipped: $skipped files" -ForegroundColor Yellow
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# List final files
Write-Host "Current files in $DataDir:" -ForegroundColor Cyan
Get-ChildItem -Path $DataDir -Filter "*.csv" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

Write-Host ""
Write-Host "✅ Done! You can now run pattern tests." -ForegroundColor Green
Write-Host ""
Write-Host "Example:" -ForegroundColor Yellow
Write-Host "  python signal_generation\tests\test_pattern.py --pattern doji --data-dir $DataDir" -ForegroundColor Gray
Write-Host ""

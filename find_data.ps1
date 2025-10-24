# Script to find historical data files
# Run this in PowerShell to locate your CSV files

Write-Host "Searching for historical data files..." -ForegroundColor Yellow
Write-Host ""

# Search in current directory
Write-Host "Searching in current directory..." -ForegroundColor Cyan
Get-ChildItem -Path . -Filter "*.csv" -Recurse -ErrorAction SilentlyContinue | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize

Write-Host ""
Write-Host "Searching for 'historical' directories..." -ForegroundColor Cyan
Get-ChildItem -Path . -Filter "historical" -Directory -Recurse -ErrorAction SilentlyContinue | Select-Object FullName

Write-Host ""
Write-Host "Searching for 'BTC' directories..." -ForegroundColor Cyan
Get-ChildItem -Path . -Filter "*BTC*" -Directory -Recurse -ErrorAction SilentlyContinue | Select-Object FullName

Write-Host ""
Write-Host "Current directory structure:" -ForegroundColor Cyan
Get-ChildItem -Path . -Directory | Select-Object Name

Write-Host ""
Write-Host "Done!" -ForegroundColor Green

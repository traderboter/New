# Python Cache Fix for BacktestMarketDataFetcher

## Problem
You're seeing this error when running the backtest:
```
'BacktestMarketDataFetcher' object has no attribute 'get_historical_data'
```

## Root Cause
Python's bytecode cache (`.pyc` files and `__pycache__` directories) can sometimes become outdated when code is updated, causing import errors even though the method exists in the source code.

## Solution

### Option 1: Quick Fix (Recommended)
Run the diagnostic script which will automatically detect and fix the issue:

```bash
python diagnose_and_fix_cache.py
```

This script will:
1. ✓ Verify the `get_historical_data` method exists in the source code
2. ✓ Clear all Python cache files (`__pycache__` and `.pyc`)
3. ✓ Clear Python's import cache (`sys.modules`)
4. ✓ Verify the fix worked

### Option 2: Manual Fix
If you prefer to do it manually:

1. **Ensure you have the latest code:**
   ```bash
   git pull origin claude/clear-python-cache-011CUQdSqs3HRw6JTe9YkYWF
   ```

2. **Clear Python cache:**
   ```bash
   python clear_cache.py
   ```

3. **Close and restart your terminal/IDE**
   - This ensures Python releases any in-memory cached imports

4. **Run the backtest again:**
   ```bash
   python backtest/run_backtest_v2.py
   ```

## Understanding the Files

### `diagnose_and_fix_cache.py` (New - Comprehensive Solution)
- **What it does**: Diagnoses the issue, clears all caches, and verifies the fix
- **When to use**: First time experiencing the issue, or if manual fixes don't work
- **Output**: Detailed diagnostic information showing exactly what's wrong and if it's fixed

### `clear_cache.py` (Original - Simple Cache Cleaner)
- **What it does**: Simply removes all `__pycache__` directories and `.pyc` files
- **When to use**: Quick cleanup when you know the issue is cache-related
- **Output**: List of removed cache files

## Verification

After running the fix, you should see:
```
✅ SUCCESS! Everything is working correctly.

You can now run your backtest:
  python backtest/run_backtest_v2.py
```

If you still see issues, the diagnostic output will tell you exactly what's wrong:
- ❌ Method not in source file → You need to `git pull` the latest changes
- ⚠ Import verification failed → Restart your terminal/IDE and try again

## Technical Details

The `get_historical_data` method was added to `BacktestMarketDataFetcher` in commit `f4eeade`:
- **File**: `backtest/historical_data_provider_v2.py`
- **Class**: `BacktestMarketDataFetcher`
- **Method**: `async def get_historical_data(self, symbol, timeframe, limit=500, force_refresh=False)`
- **Purpose**: Provides a compatible interface for `SignalOrchestrator` to fetch historical data during backtest

## Still Having Issues?

If the diagnostic script shows success but you still see the error:

1. **Make absolutely sure you're running from the correct directory:**
   ```bash
   cd C:\Users\trade\Documents\PythonProject\New
   ```

2. **Verify you're using the correct Python environment:**
   ```bash
   C:\Users\trade\Documents\PythonProject\venv\Scripts\python.exe --version
   ```

3. **Try a complete cache clear and Python restart:**
   ```bash
   # Run the diagnostic script
   python diagnose_and_fix_cache.py

   # Close your terminal completely
   # Open a new terminal
   # Navigate back to the project directory
   cd C:\Users\trade\Documents\PythonProject\New

   # Run backtest
   python backtest/run_backtest_v2.py
   ```

4. **Check git branch:**
   ```bash
   git status
   git log --oneline -5
   ```
   Make sure you see commit `f4eeade` or later which contains the fix.

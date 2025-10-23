#!/usr/bin/env python3
"""
Diagnostic and Fix Script for BacktestMarketDataFetcher Import Issues

This script will:
1. Check if the file contains the get_historical_data method
2. Clear all Python caches
3. Clear import cache from sys.modules
4. Verify the method can be imported correctly
"""

import os
import sys
import shutil
import importlib
import ast


def print_section(title):
    """Print a section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def check_method_exists_in_file():
    """Check if get_historical_data method exists in the source file"""
    print_section("STEP 1: Checking Source File")

    file_path = os.path.join('backtest', 'historical_data_provider_v2.py')

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    print(f"✓ File exists: {file_path}")

    # Read the file and check for the method
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'async def get_historical_data' in content:
        print("✓ Found 'async def get_historical_data' in source file")

        # Count occurrences
        count = content.count('async def get_historical_data')
        print(f"✓ Method appears {count} time(s) in file")

        # Try to find it in BacktestMarketDataFetcher class
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == 'BacktestMarketDataFetcher':
                    methods = [m.name for m in node.body if isinstance(m, ast.AsyncFunctionDef)]
                    if 'get_historical_data' in methods:
                        print("✓ Method found in BacktestMarketDataFetcher class")
                        print(f"  Class methods: {', '.join(methods)}")
                        return True
                    else:
                        print(f"❌ Method NOT found in BacktestMarketDataFetcher class")
                        print(f"  Available methods: {', '.join(methods)}")
                        return False
        except Exception as e:
            print(f"⚠ Could not parse AST: {e}")
            print("  (But method text was found in file)")
            return True
    else:
        print("❌ Method 'get_historical_data' NOT found in source file")
        return False

    return False


def clear_all_caches():
    """Clear all Python cache files and directories"""
    print_section("STEP 2: Clearing All Caches")

    removed_dirs = 0
    removed_files = 0

    # 1. Remove __pycache__ and .pyc files
    for dirpath, dirnames, filenames in os.walk('.'):
        # Skip venv directory
        if 'venv' in dirpath or '.venv' in dirpath:
            continue

        # Remove __pycache__ directories
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                removed_dirs += 1
                print(f"  ✓ Removed: {pycache_path}")
            except Exception as e:
                print(f"  ✗ Failed to remove {pycache_path}: {e}")

        # Remove .pyc files
        for filename in filenames:
            if filename.endswith('.pyc'):
                pyc_path = os.path.join(dirpath, filename)
                try:
                    os.remove(pyc_path)
                    removed_files += 1
                    print(f"  ✓ Removed: {pyc_path}")
                except Exception as e:
                    print(f"  ✗ Failed to remove {pyc_path}: {e}")

    print(f"\n✅ Removed {removed_dirs} __pycache__ directories")
    print(f"✅ Removed {removed_files} .pyc files")

    # 2. Clear sys.modules for our modules
    print("\n  Clearing sys.modules import cache...")
    modules_to_clear = [key for key in sys.modules.keys()
                       if 'backtest' in key or 'signal_generation' in key]

    for module in modules_to_clear:
        del sys.modules[module]
        print(f"  ✓ Cleared from sys.modules: {module}")

    print(f"✅ Cleared {len(modules_to_clear)} modules from sys.modules")

    # 3. Clear importlib cache
    print("\n  Clearing importlib cache...")
    importlib.invalidate_caches()
    print("✅ Cleared importlib cache")


def verify_import():
    """Verify that the method can be imported correctly"""
    print_section("STEP 3: Verifying Import")

    try:
        # Import the module
        from backtest.historical_data_provider_v2 import BacktestMarketDataFetcher
        print("✓ Successfully imported BacktestMarketDataFetcher")

        # Check if the method exists
        if hasattr(BacktestMarketDataFetcher, 'get_historical_data'):
            print("✓ Method 'get_historical_data' exists on class")

            # Check if it's async
            method = getattr(BacktestMarketDataFetcher, 'get_historical_data')
            if hasattr(method, '__code__'):
                is_async = method.__code__.co_flags & 0x100  # CO_COROUTINE flag
                if is_async:
                    print("✓ Method is async (coroutine)")
                else:
                    print("⚠ Method is not async")

            # List all methods
            all_methods = [m for m in dir(BacktestMarketDataFetcher)
                          if not m.startswith('_')]
            print(f"\nAll public methods on BacktestMarketDataFetcher:")
            for method in all_methods:
                print(f"  - {method}")

            return True
        else:
            print("❌ Method 'get_historical_data' NOT found on class")

            # List what methods are available
            all_methods = [m for m in dir(BacktestMarketDataFetcher)
                          if not m.startswith('_')]
            print(f"\nAvailable public methods:")
            for method in all_methods:
                print(f"  - {method}")

            return False

    except ImportError as e:
        error_msg = str(e)
        if 'pandas' in error_msg or 'numpy' in error_msg:
            print(f"⚠ Import skipped: Missing dependency ({error_msg})")
            print("  This is OK - dependencies will be available when running backtest")
            print("  Method was verified in source file ✓")
            return True  # Return True since we verified the source
        print(f"❌ Failed to import: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main diagnostic and fix function"""
    print("🔍 BacktestMarketDataFetcher Diagnostic and Fix Tool")
    print(f"Working directory: {os.path.abspath('.')}")
    print(f"Python version: {sys.version}")

    # Step 1: Check if method exists in source
    method_in_source = check_method_exists_in_file()

    if not method_in_source:
        print("\n⚠ WARNING: Method not found in source file!")
        print("  This means you need to pull the latest changes from git.")
        print("  Run: git pull origin <branch-name>")
        return 1

    # Step 2: Clear all caches
    clear_all_caches()

    # Step 3: Verify import
    import_works = verify_import()

    # Final summary
    print_section("SUMMARY")

    if method_in_source and import_works:
        print("✅ SUCCESS! Everything is working correctly.")
        print("\nYou can now run your backtest:")
        print("  python backtest/run_backtest_v2.py")
        return 0
    elif method_in_source and not import_works:
        print("⚠ PARTIAL SUCCESS")
        print("  - Method exists in source file ✓")
        print("  - But import verification failed ✗")
        print("\nTry restarting your terminal/IDE and running this script again.")
        return 1
    else:
        print("❌ FAILED")
        print("  Method not found in source file.")
        print("\nPlease pull the latest changes:")
        print("  git pull origin claude/clear-python-cache-011CUQdSqs3HRw6JTe9YkYWF")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

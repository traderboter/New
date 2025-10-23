#!/usr/bin/env python3
"""
Clear Python Cache Script
Removes all __pycache__ directories and .pyc files to resolve import issues
"""

import os
import shutil
import sys


def clear_cache(root_dir='.'):
    """
    Clear all Python cache files and directories.

    Args:
        root_dir: Root directory to start searching from (default: current directory)
    """
    removed_dirs = 0
    removed_files = 0

    print(f"🧹 Clearing Python cache in: {os.path.abspath(root_dir)}")
    print("-" * 60)

    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
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

    print("-" * 60)
    print(f"✅ Cache clearing complete!")
    print(f"   - Removed {removed_dirs} __pycache__ directories")
    print(f"   - Removed {removed_files} .pyc files")
    print()
    print("💡 You can now run your backtest again.")


if __name__ == "__main__":
    # Get root directory from command line or use current directory
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    clear_cache(root)

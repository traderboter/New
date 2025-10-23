"""
اسکریپت ساده برای اجرای Backtest V2

🆕 نسخه 2.0 با SignalOrchestrator
"""

import asyncio
import logging
import sys
from pathlib import Path

# اضافه کردن root به path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.backtest_engine_v2 import run_backtest_v2

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    try:
        print("=" * 70)
        print(" " * 20 + "🚀 BACKTEST V2.0")
        print(" " * 15 + "with SignalOrchestrator")
        print("=" * 70)
        
        engine, results_dir = asyncio.run(
            run_backtest_v2('backtest/config_backtest_v2.yaml')
        )
        
        print(f"\n✅ Backtest V2 completed successfully!")
        print(f"📁 Results saved to: {results_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

"""
تست سریالیزیشن Trade با فیلدهای pattern tracking

این اسکریپت تست می‌کند که آیا Trade.to_dict() و Trade.from_dict()
فیلدهای جدید pattern tracking را صحیح ذخیره و بازیابی می‌کنند.
"""

import sys
sys.path.insert(0, '/home/user/New')

from datetime import datetime
import json

# Import directly to avoid dependencies
import importlib.util

spec = importlib.util.spec_from_file_location("multi_tp_trade", "/home/user/New/multi_tp_trade.py")
multi_tp_trade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_tp_trade)
Trade = multi_tp_trade.Trade


def test_trade_serialization():
    """تست سریالیزیشن و دیسریالیزیشن Trade"""

    print("="*80)
    print("🧪 تست سریالیزیشن Trade با فیلدهای Pattern Tracking")
    print("="*80)

    # نمونه اطلاعات الگوها
    signal_patterns_details = [
        {
            'name': 'Hammer',
            'type': 'candlestick',
            'direction': 'bullish',
            'timeframe': '1h',
            'candles_ago': 2,
            'recency_multiplier': 0.8,
            'base_strength': 2,
            'adjusted_strength': 1.6,
            'confidence': 0.75
        },
        {
            'name': 'MACD Bullish Cross',
            'type': 'indicator',
            'direction': 'bullish',
            'timeframe': '1h',
            'candles_ago': 0,
            'recency_multiplier': 1.0,
            'base_strength': 2,
            'adjusted_strength': 2.0,
            'confidence': 0.82
        }
    ]

    signal_pattern_contributions = {
        'Hammer': 15.2,
        'MACD Bullish Cross': 12.8,
        'RSI_oversold': 8.5
    }

    signal_score_breakdown = {
        'base_scores': {'trend': 75.3, 'momentum': 68.2},
        'weighted_scores': {'trend': 22.59, 'momentum': 17.05},
        'aggregates': {
            'base_score': 61.91,
            'confluence_bonus': 0.3,
            'timeframe_weight': 1.2
        },
        'final': {'score': 92.4, 'confidence': 0.82, 'strength': 'strong'}
    }

    # ایجاد Trade
    print("\n1️⃣  ایجاد Trade با فیلدهای pattern tracking...")
    trade = Trade(
        trade_id='TEST_001',
        symbol='BTC/USDT',
        direction='long',
        entry_price=67500.0,
        stop_loss=66800.0,
        take_profit=69200.0,
        quantity=0.1,
        risk_amount=100.0,
        timestamp=datetime.now(),
        status='open',
        signal_patterns_details=signal_patterns_details,
        signal_pattern_contributions=signal_pattern_contributions,
        signal_score_breakdown=signal_score_breakdown
    )

    print(f"   ✅ Trade ایجاد شد: {trade.trade_id}")
    print(f"   ✅ تعداد الگوها: {len(trade.signal_patterns_details)}")
    print(f"   ✅ تعداد contributions: {len(trade.signal_pattern_contributions)}")
    print(f"   ✅ breakdown موجود: {bool(trade.signal_score_breakdown)}")

    # سریالیز کردن
    print("\n2️⃣  سریالیزیشن با to_dict()...")
    trade_dict = trade.to_dict()

    # بررسی وجود فیلدها در dict
    assert 'signal_patterns_details' in trade_dict, "❌ signal_patterns_details در dict نیست!"
    assert 'signal_pattern_contributions' in trade_dict, "❌ signal_pattern_contributions در dict نیست!"
    assert 'signal_score_breakdown' in trade_dict, "❌ signal_score_breakdown در dict نیست!"

    print(f"   ✅ تمام فیلدها در dict موجود هستند")
    print(f"   ✅ signal_patterns_details: {len(trade_dict['signal_patterns_details'])} الگو")
    print(f"   ✅ signal_pattern_contributions: {len(trade_dict['signal_pattern_contributions'])} مورد")

    # تست JSON serialization (مهم برای database)
    print("\n3️⃣  تست JSON serialization (برای database)...")
    try:
        json_str = json.dumps(trade_dict, default=str, ensure_ascii=False)
        print(f"   ✅ JSON serialization موفق - طول: {len(json_str)} کاراکتر")

        # بررسی که اطلاعات الگوها در JSON هست
        assert 'Hammer' in json_str, "❌ اطلاعات Hammer در JSON نیست!"
        assert 'MACD Bullish Cross' in json_str, "❌ اطلاعات MACD در JSON نیست!"
        print(f"   ✅ اطلاعات الگوها در JSON موجود است")
    except Exception as e:
        print(f"   ❌ خطا در JSON serialization: {e}")
        raise

    # دیسریالیز کردن
    print("\n4️⃣  دیسریالیزیشن با from_dict()...")
    trade_restored = Trade.from_dict(trade_dict)

    print(f"   ✅ Trade بازیابی شد: {trade_restored.trade_id}")

    # بررسی فیلدهای الگو
    assert hasattr(trade_restored, 'signal_patterns_details'), "❌ trade_restored فیلد signal_patterns_details ندارد!"
    assert hasattr(trade_restored, 'signal_pattern_contributions'), "❌ trade_restored فیلد signal_pattern_contributions ندارد!"
    assert hasattr(trade_restored, 'signal_score_breakdown'), "❌ trade_restored فیلد signal_score_breakdown ندارد!"

    print(f"   ✅ تمام فیلدها در trade بازیابی شده موجود هستند")

    # بررسی دقیق محتوا
    print("\n5️⃣  بررسی دقت بازیابی...")

    assert len(trade_restored.signal_patterns_details) == len(signal_patterns_details), \
        f"❌ تعداد الگوها مطابقت ندارد: {len(trade_restored.signal_patterns_details)} != {len(signal_patterns_details)}"
    print(f"   ✅ تعداد الگوها مطابقت دارد: {len(trade_restored.signal_patterns_details)}")

    assert len(trade_restored.signal_pattern_contributions) == len(signal_pattern_contributions), \
        f"❌ تعداد contributions مطابقت ندارد"
    print(f"   ✅ تعداد contributions مطابقت دارد: {len(trade_restored.signal_pattern_contributions)}")

    # بررسی مقادیر
    assert trade_restored.signal_patterns_details[0]['name'] == 'Hammer', "❌ نام الگوی اول مطابقت ندارد!"
    assert trade_restored.signal_pattern_contributions['Hammer'] == 15.2, "❌ سهم Hammer مطابقت ندارد!"
    assert trade_restored.signal_score_breakdown['final']['score'] == 92.4, "❌ امتیاز نهایی مطابقت ندارد!"

    print(f"   ✅ نام اولین الگو: {trade_restored.signal_patterns_details[0]['name']}")
    print(f"   ✅ سهم Hammer: {trade_restored.signal_pattern_contributions['Hammer']}")
    print(f"   ✅ امتیاز نهایی: {trade_restored.signal_score_breakdown['final']['score']}")

    print("\n" + "="*80)
    print("✅ تمام تست‌های سریالیزیشن موفق بودند!")
    print("="*80)

    print("\n💡 نتیجه:")
    print("   📊 فیلدهای pattern tracking به درستی در to_dict() ذخیره می‌شوند")
    print("   📊 فیلدهای pattern tracking به درستی از from_dict() بازیابی می‌شوند")
    print("   📊 JSON serialization برای database کار می‌کند")
    print("   ✅ معاملات با تمام جزئیات الگوها در database ذخیره خواهند شد")

    return trade, trade_restored


if __name__ == "__main__":
    try:
        test_trade_serialization()
    except Exception as e:
        print(f"\n❌ خطا در تست: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
تست سیستم جدید ردیابی الگوها در TradeResult

این اسکریپت نمونه‌ای از نحوه استفاده از فیلدهای جدید TradeResult را نشان می‌دهد.
"""

import sys
sys.path.insert(0, '/home/user/New')

from datetime import datetime, timezone, timedelta

# Import directly from file to avoid pandas dependency
import importlib.util
spec = importlib.util.spec_from_file_location("data_models", "/home/user/New/signal_generation/shared/data_models.py")
data_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_models)
TradeResult = data_models.TradeResult

def test_trade_result_with_patterns():
    """تست ایجاد TradeResult با جزئیات الگوها"""

    print("="*80)
    print("🧪 تست سیستم ردیابی الگوها در TradeResult")
    print("="*80)

    # نمونه اطلاعات الگوهای تشخیص داده شده
    detected_patterns = [
        {
            'name': 'Hammer',
            'type': 'candlestick',
            'direction': 'bullish',
            'timeframe': '1h',
            'candles_ago': 2,
            'recency_multiplier': 0.8,
            'base_strength': 2,
            'adjusted_strength': 1.6,
            'confidence': 0.75,
            'metadata': {
                'quality_score': 85.5,
                'hammer_type': 'Strong'
            }
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
        },
        {
            'name': 'Double Bottom',
            'type': 'chart',
            'direction': 'bullish',
            'timeframe': '4h',
            'candles_ago': 1,
            'recency_multiplier': 0.9,
            'base_strength': 3,
            'adjusted_strength': 2.7,
            'confidence': 0.88
        }
    ]

    # سهم هر الگو در امتیاز نهایی
    pattern_contributions = {
        'Hammer': 15.2,
        'MACD Bullish Cross': 12.8,
        'Double Bottom': 22.5,
        'RSI_oversold': 8.5
    }

    # breakdown کامل امتیاز
    score_breakdown = {
        'base_scores': {
            'trend': 75.3,
            'momentum': 68.2,
            'volume': 82.1,
            'pattern': 58.5
        },
        'weighted_scores': {
            'trend': 22.59,
            'momentum': 17.05,
            'volume': 16.42,
            'pattern': 5.85
        },
        'aggregates': {
            'base_score': 61.91,
            'confluence_bonus': 0.3,
            'timeframe_weight': 1.2,
            'htf_multiplier': 1.3,
            'volatility_multiplier': 0.95
        },
        'final': {
            'score': 92.4,
            'confidence': 0.82,
            'strength': 'strong'
        }
    }

    # ایجاد TradeResult
    entry_time = datetime.now(timezone.utc) - timedelta(hours=3)
    exit_time = datetime.now(timezone.utc)

    trade_result = TradeResult(
        signal_id='BTC_L_1730000000_1234',
        symbol='BTC/USDT',
        direction='long',
        entry_price=67500.0,
        exit_price=69200.0,
        stop_loss=66800.0,
        take_profit=69200.0,
        entry_time=entry_time,
        exit_time=exit_time,
        exit_reason='take_profit_hit',
        profit_pct=2.52,
        profit_r=2.43,
        market_regime='trending',
        pattern_names=['Hammer', 'MACD Bullish Cross', 'Double Bottom'],  # backward compatibility
        timeframe='1h',
        signal_score=92.4,
        signal_type='multi_timeframe',
        # 🆕 فیلدهای جدید
        detected_patterns_details=detected_patterns,
        pattern_contributions=pattern_contributions,
        score_breakdown=score_breakdown
    )

    print("\n✅ TradeResult با موفقیت ایجاد شد!")
    print(f"\n📊 اطلاعات اصلی معامله:")
    print(f"   نماد: {trade_result.symbol}")
    print(f"   جهت: {trade_result.direction}")
    print(f"   ورود: {trade_result.entry_price:.2f}")
    print(f"   خروج: {trade_result.exit_price:.2f}")
    print(f"   سود: {trade_result.profit_pct:.2f}% ({trade_result.profit_r:.2f}R)")
    print(f"   مدت: {trade_result.trade_duration}")

    print(f"\n🎯 الگوهای تشخیص داده شده ({len(trade_result.detected_patterns_details)} الگو):")
    for i, pattern in enumerate(trade_result.detected_patterns_details, 1):
        print(f"   {i}. {pattern['name']} [{pattern['timeframe']}]")
        print(f"      - نوع: {pattern['type']}, جهت: {pattern['direction']}")
        print(f"      - تشکیل در کندل: {pattern['candles_ago']} کندل قبل")
        print(f"      - ضریب recency: {pattern['recency_multiplier']}")
        print(f"      - قدرت: {pattern['base_strength']} → {pattern['adjusted_strength']} (بعد از تنظیم)")
        print(f"      - اعتماد: {pattern['confidence']:.2%}")

    print(f"\n💯 سهم هر الگو در امتیاز نهایی:")
    for pattern_name, contribution in trade_result.pattern_contributions.items():
        print(f"   {pattern_name}: {contribution:.2f}")

    print(f"\n📈 breakdown امتیاز:")
    print(f"   امتیاز پایه: {trade_result.score_breakdown['aggregates']['base_score']:.2f}")
    print(f"   بونوس همگرایی: {trade_result.score_breakdown['aggregates']['confluence_bonus']:.2f}")
    print(f"   وزن تایم‌فریم: {trade_result.score_breakdown['aggregates']['timeframe_weight']:.2f}")
    print(f"   ضریب HTF: {trade_result.score_breakdown['aggregates']['htf_multiplier']:.2f}")
    print(f"   ضریب نوسان: {trade_result.score_breakdown['aggregates']['volatility_multiplier']:.2f}")
    print(f"   ➜ امتیاز نهایی: {trade_result.score_breakdown['final']['score']:.2f}")
    print(f"   ➜ اعتماد: {trade_result.score_breakdown['final']['confidence']:.2%}")
    print(f"   ➜ قدرت: {trade_result.score_breakdown['final']['strength']}")

    # تست سریالیزیشن
    print(f"\n🔄 تست سریالیزیشن...")
    trade_dict = trade_result.to_dict()
    print(f"   ✅ to_dict() موفقیت‌آمیز - {len(trade_dict)} فیلد")

    trade_restored = TradeResult.from_dict(trade_dict)
    print(f"   ✅ from_dict() موفقیت‌آمیز")
    print(f"   ✅ الگوها بازیابی شدند: {len(trade_restored.detected_patterns_details)} الگو")
    print(f"   ✅ contributions بازیابی شدند: {len(trade_restored.pattern_contributions)} مورد")

    print("\n" + "="*80)
    print("✅ تمام تست‌ها با موفقیت انجام شدند!")
    print("="*80)

    return trade_result


if __name__ == "__main__":
    try:
        trade_result = test_trade_result_with_patterns()

        print("\n💡 نتیجه‌گیری:")
        print("   با این سیستم جدید، شما می‌توانید دقیقاً بدانید که:")
        print("   1️⃣  هر معامله به خاطر تشکیل چه الگوهایی انجام شده")
        print("   2️⃣  هر الگو در کدام تایم‌فریم تشکیل شده")
        print("   3️⃣  هر الگو در کندل چندم تشکیل شده (recency)")
        print("   4️⃣  هر الگو چقدر به امتیاز نهایی کمک کرده")
        print("   5️⃣  امتیاز نهایی دقیقاً چگونه محاسبه شده")

    except Exception as e:
        print(f"\n❌ خطا در اجرای تست: {e}")
        import traceback
        traceback.print_exc()

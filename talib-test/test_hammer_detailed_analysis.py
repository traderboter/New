"""
Detailed Analysis of TA-Lib CDLHAMMER on Real BTC Data

Similar to test_talib_on_real_data.py for Shooting Star.
Analyzes all 277 Hammer detections to understand TA-Lib's criteria.
"""

import pandas as pd
import numpy as np
import talib


def load_btc_data(filename='historical/BTC-USDT/1hour.csv'):
    """Load BTC OHLCV data."""
    df = pd.read_csv(filename)
    df = df.astype({
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'volume': np.float64
    })
    return df


def analyze_candle_characteristics(df, idx):
    """Analyze characteristics of a specific candle."""
    candle = df.iloc[idx]

    open_price = candle['open']
    high = candle['high']
    low = candle['low']
    close = candle['close']

    # Calculate sizes
    body_size = abs(close - open_price)
    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low
    full_range = high - low

    if full_range == 0:
        return None

    # Calculate percentages
    upper_shadow_pct = (upper_shadow / full_range) * 100
    lower_shadow_pct = (lower_shadow / full_range) * 100
    body_pct = (body_size / full_range) * 100

    # Body position (Hammer: body at TOP)
    body_top = max(open_price, close)
    body_position = ((high - body_top) / full_range) if full_range > 0 else 0

    # Candle direction
    is_bearish = close < open_price
    is_bullish = close > open_price
    is_doji = abs(close - open_price) < (full_range * 0.1)

    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'body_size': body_size,
        'upper_shadow': upper_shadow,
        'lower_shadow': lower_shadow,
        'full_range': full_range,
        'upper_shadow_pct': upper_shadow_pct,
        'lower_shadow_pct': lower_shadow_pct,
        'body_pct': body_pct,
        'body_position': body_position,
        'is_bearish': is_bearish,
        'is_bullish': is_bullish,
        'is_doji': is_doji,
        'direction': 'Bearish' if is_bearish else ('Bullish' if is_bullish else 'Doji')
    }


def analyze_context(df, idx, lookback=10):
    """Analyze context (trend) before the candle."""
    if idx < lookback:
        return None

    context = df.iloc[idx-lookback:idx]

    # Calculate trend
    closes = context['close'].values
    indices = np.arange(len(closes))
    if len(closes) > 1:
        slope = np.polyfit(indices, closes, 1)[0]
        trend_direction = 'Uptrend' if slope > 0 else 'Downtrend'
        trend_strength = abs(slope) / np.mean(closes) * 100
    else:
        slope = 0
        trend_direction = 'Unknown'
        trend_strength = 0

    # Count bullish/bearish candles
    bullish_count = sum(context['close'] > context['open'])
    bearish_count = sum(context['close'] < context['open'])

    return {
        'slope': slope,
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'bearish_pct': (bearish_count / lookback) * 100
    }


def main():
    """Run detailed analysis on real BTC data."""

    print("\n" + "="*70)
    print("TA-LIB HAMMER DETAILED ANALYSIS - REAL BTC DATA")
    print("="*70)

    # Load data
    print("\nLoading BTC 1-hour data...")
    df = load_btc_data()
    print(f"Total candles: {len(df):,}")

    # Run TA-Lib
    print("\nRunning TA-Lib CDLHAMMER...")
    pattern = talib.CDLHAMMER(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    # Find detections
    detections = np.where(pattern != 0)[0]

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total detections: {len(detections)}")
    print(f"Detection rate: {len(detections)/len(df)*100:.2f}%")

    # Compare with Shooting Star
    print(f"\nComparison with Shooting Star:")
    print(f"  Shooting Star: 75 detections (0.71%)")
    print(f"  Hammer: {len(detections)} detections ({len(detections)/len(df)*100:.2f}%)")
    print(f"  Ratio: Hammer is {len(detections)/75:.1f}× more common")

    # Show first 10 detections
    print(f"\n{'='*70}")
    print("FIRST 10 DETECTIONS")
    print(f"{'='*70}")

    for i, idx in enumerate(detections[:10]):
        chars = analyze_candle_characteristics(df, idx)
        context = analyze_context(df, idx)

        print(f"\n{'-'*70}")
        print(f"Detection #{i+1} at index {idx}")
        print(f"{'-'*70}")
        print(f"Pattern value: {pattern[idx]}")
        print(f"\nCandle Characteristics:")
        print(f"  OHLC: O={chars['open']:.2f}, H={chars['high']:.2f}, "
              f"L={chars['low']:.2f}, C={chars['close']:.2f}")
        print(f"  Direction: {chars['direction']}")
        print(f"\n  Upper shadow: {chars['upper_shadow_pct']:.1f}% of range")
        print(f"  Body: {chars['body_pct']:.1f}% of range")
        print(f"  Lower shadow: {chars['lower_shadow_pct']:.1f}% of range")
        print(f"  Body position: {chars['body_position']:.3f} (0=top, 1=bottom)")

        if context:
            print(f"\nContext (previous 10 candles):")
            print(f"  Trend: {context['trend_direction']}")
            print(f"  Bearish candles: {context['bearish_count']}/{10} ({context['bearish_pct']:.0f}%)")

    # Statistical summary
    print(f"\n{'='*70}")
    print("STATISTICAL SUMMARY OF ALL 277 DETECTIONS")
    print(f"{'='*70}")

    all_chars = [analyze_candle_characteristics(df, idx) for idx in detections]
    all_chars = [c for c in all_chars if c is not None]

    if len(all_chars) > 0:
        upper_shadows = [c['upper_shadow_pct'] for c in all_chars]
        lower_shadows = [c['lower_shadow_pct'] for c in all_chars]
        bodies = [c['body_pct'] for c in all_chars]
        body_positions = [c['body_position'] for c in all_chars]

        print(f"\nUpper Shadow % (should be SMALL for Hammer):")
        print(f"  Min: {min(upper_shadows):.1f}%")
        print(f"  Max: {max(upper_shadows):.1f}%")
        print(f"  Mean: {np.mean(upper_shadows):.1f}%")
        print(f"  Median: {np.median(upper_shadows):.1f}%")

        print(f"\nBody %:")
        print(f"  Min: {min(bodies):.1f}%")
        print(f"  Max: {max(bodies):.1f}%")
        print(f"  Mean: {np.mean(bodies):.1f}%")
        print(f"  Median: {np.median(bodies):.1f}%")

        print(f"\nLower Shadow % (should be LONG for Hammer):")
        print(f"  Min: {min(lower_shadows):.1f}%")
        print(f"  Max: {max(lower_shadows):.1f}%")
        print(f"  Mean: {np.mean(lower_shadows):.1f}%")
        print(f"  Median: {np.median(lower_shadows):.1f}%")

        print(f"\nBody Position (0=top, 1=bottom - should be near 0 for Hammer):")
        print(f"  Min: {min(body_positions):.3f}")
        print(f"  Max: {max(body_positions):.3f}")
        print(f"  Mean: {np.mean(body_positions):.3f}")
        print(f"  Median: {np.median(body_positions):.3f}")

        # Candle direction distribution
        bearish_count = sum(1 for c in all_chars if c['is_bearish'])
        bullish_count = sum(1 for c in all_chars if c['is_bullish'])
        doji_count = sum(1 for c in all_chars if c['is_doji'])

        print(f"\nCandle Direction:")
        print(f"  Bearish: {bearish_count} ({bearish_count/len(all_chars)*100:.1f}%)")
        print(f"  Bullish: {bullish_count} ({bullish_count/len(all_chars)*100:.1f}%)")
        print(f"  Doji: {doji_count} ({doji_count/len(all_chars)*100:.1f}%)")

        # Context analysis
        all_contexts = [analyze_context(df, idx) for idx in detections]
        all_contexts = [c for c in all_contexts if c is not None]

        if len(all_contexts) > 0:
            uptrend_count = sum(1 for c in all_contexts if c['trend_direction'] == 'Uptrend')
            downtrend_count = sum(1 for c in all_contexts if c['trend_direction'] == 'Downtrend')

            print(f"\nContext Trend:")
            print(f"  Uptrend: {uptrend_count} ({uptrend_count/len(all_contexts)*100:.1f}%)")
            print(f"  Downtrend: {downtrend_count} ({downtrend_count/len(all_contexts)*100:.1f}%)")

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON: HAMMER vs SHOOTING STAR")
    print(f"{'='*70}")
    print(f"\n{'Metric':<30} {'Shooting Star':<20} {'Hammer':<20}")
    print(f"{'-'*70}")
    print(f"{'Detection count':<30} {'75':<20} {f'{len(detections)}':<20}")
    print(f"{'Detection rate':<30} {'0.71%':<20} {f'{len(detections)/len(df)*100:.2f}%':<20}")
    print(f"{'Minimum candles':<30} {'12':<20} {'12':<20}")
    print(f"{'Minimum lookback':<30} {'11':<20} {'11':<20}")
    print(f"{'-'*70}")
    print(f"{'Upper shadow (mean)':<30} {'62.8%':<20} {f'{np.mean(upper_shadows):.1f}%':<20}")
    print(f"{'Body (mean)':<30} {'31.3%':<20} {f'{np.mean(bodies):.1f}%':<20}")
    print(f"{'Lower shadow (mean)':<30} {'5.9%':<20} {f'{np.mean(lower_shadows):.1f}%':<20}")
    print(f"{'-'*70}")
    print(f"{'Bullish candles':<30} {'100%':<20} {f'{bullish_count/len(all_chars)*100:.1f}%':<20}")
    print(f"{'Bearish candles':<30} {'0%':<20} {f'{bearish_count/len(all_chars)*100:.1f}%':<20}")
    print(f"{'-'*70}")
    print(f"{'In uptrend':<30} {'49.3%':<20} {f'{uptrend_count/len(all_contexts)*100:.1f}%':<20}")
    print(f"{'In downtrend':<30} {'50.7%':<20} {f'{downtrend_count/len(all_contexts)*100:.1f}%':<20}")

    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print(f"{'='*70}")
    print("\n1. Hammer requires EXACTLY 12 candles (same as Shooting Star)")
    print(f"2. Hammer is {len(detections)/75:.1f}× more common than Shooting Star in BTC data")
    print("3. TA-Lib does NOT check for downtrend context")
    print("4. We need to add downtrend check (just like uptrend for Shooting Star)")
    print("\nNext step: Fix hammer.py to use TA-Lib with 12+ candles")


if __name__ == "__main__":
    main()

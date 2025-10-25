"""
Test TA-Lib CDLSHOOTINGSTAR on REAL BTC data

Goal: Find which candles TA-Lib actually detects as Shooting Star
and analyze their characteristics
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

    # Calculate percentages relative to full range
    upper_shadow_pct = (upper_shadow / full_range) * 100
    lower_shadow_pct = (lower_shadow / full_range) * 100
    body_pct = (body_size / full_range) * 100

    # Body position
    body_bottom = min(open_price, close)
    body_position = ((body_bottom - low) / full_range) if full_range > 0 else 0

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
        'bullish_pct': (bullish_count / lookback) * 100
    }


def main():
    """Run analysis on real BTC data."""

    print("\n" + "="*70)
    print("TA-LIB SHOOTING STAR ON REAL BTC DATA")
    print("="*70)

    # Load data
    print("\nLoading BTC 1-hour data...")
    df = load_btc_data()
    print(f"Total candles: {len(df):,}")

    # Run TA-Lib
    print("\nRunning TA-Lib CDLSHOOTINGSTAR...")
    pattern = talib.CDLSHOOTINGSTAR(
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

    if len(detections) == 0:
        print("\n❌ TA-Lib found NO Shooting Star patterns in BTC data!")
        print("\nThis is VERY interesting - let's investigate why...")

        # Sample some candles and show why they weren't detected
        print("\n" + "-"*70)
        print("Sample candles that LOOK like Shooting Star but weren't detected:")
        print("-"*70)

        # Find candles with long upper shadow and small body
        df['body_pct'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['upper_shadow_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
        df['lower_shadow_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])

        # Filter: long upper shadow, small body, small lower shadow
        candidates = df[
            (df['upper_shadow_pct'] >= 0.5) &
            (df['body_pct'] <= 0.3) &
            (df['lower_shadow_pct'] <= 0.2) &
            (df['high'] - df['low'] > 0)
        ]

        print(f"\nFound {len(candidates)} candles matching Shooting Star physics")
        print("(upper_shadow >= 50%, body <= 30%, lower_shadow <= 20%)")

        if len(candidates) > 0:
            print("\nShowing first 5 candidates:\n")
            for i, (idx, row) in enumerate(candidates.head(5).iterrows()):
                chars = analyze_candle_characteristics(df, idx)
                context = analyze_context(df, idx)

                print(f"\nCandidate #{i+1} (index {idx}):")
                print(f"  OHLC: O={chars['open']:.2f}, H={chars['high']:.2f}, "
                      f"L={chars['low']:.2f}, C={chars['close']:.2f}")
                print(f"  Direction: {chars['direction']}")
                print(f"  Upper shadow: {chars['upper_shadow_pct']:.1f}%")
                print(f"  Body: {chars['body_pct']:.1f}%")
                print(f"  Lower shadow: {chars['lower_shadow_pct']:.1f}%")
                print(f"  Body position: {chars['body_position']:.3f}")
                if context:
                    print(f"  Context: {context['trend_direction']} ({context['bullish_pct']:.0f}% bullish candles)")

    else:
        print(f"\n✅ Found {len(detections)} Shooting Star patterns!")

        # Analyze each detection
        print("\n" + "="*70)
        print("DETAILED ANALYSIS OF EACH DETECTION")
        print("="*70)

        for i, idx in enumerate(detections[:20]):  # Show first 20
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
            print(f"  Full range: ${chars['full_range']:.2f}")
            print(f"\n  Upper shadow: ${chars['upper_shadow']:.2f} ({chars['upper_shadow_pct']:.1f}% of range)")
            print(f"  Body: ${chars['body_size']:.2f} ({chars['body_pct']:.1f}% of range)")
            print(f"  Lower shadow: ${chars['lower_shadow']:.2f} ({chars['lower_shadow_pct']:.1f}% of range)")
            print(f"  Body position: {chars['body_position']:.3f} (0=bottom, 1=top)")

            if context:
                print(f"\nContext (previous 10 candles):")
                print(f"  Trend: {context['trend_direction']}")
                print(f"  Trend strength: {context['trend_strength']:.2f}%")
                print(f"  Bullish candles: {context['bullish_count']}/{10} ({context['bullish_pct']:.0f}%)")
                print(f"  Bearish candles: {context['bearish_count']}/{10}")

        if len(detections) > 20:
            print(f"\n... and {len(detections) - 20} more detections")

        # Statistical summary
        print("\n" + "="*70)
        print("STATISTICAL SUMMARY")
        print("="*70)

        all_chars = [analyze_candle_characteristics(df, idx) for idx in detections]
        all_chars = [c for c in all_chars if c is not None]

        if len(all_chars) > 0:
            upper_shadows = [c['upper_shadow_pct'] for c in all_chars]
            lower_shadows = [c['lower_shadow_pct'] for c in all_chars]
            bodies = [c['body_pct'] for c in all_chars]
            body_positions = [c['body_position'] for c in all_chars]

            print(f"\nUpper Shadow %:")
            print(f"  Min: {min(upper_shadows):.1f}%")
            print(f"  Max: {max(upper_shadows):.1f}%")
            print(f"  Mean: {np.mean(upper_shadows):.1f}%")
            print(f"  Median: {np.median(upper_shadows):.1f}%")

            print(f"\nBody %:")
            print(f"  Min: {min(bodies):.1f}%")
            print(f"  Max: {max(bodies):.1f}%")
            print(f"  Mean: {np.mean(bodies):.1f}%")
            print(f"  Median: {np.median(bodies):.1f}%")

            print(f"\nLower Shadow %:")
            print(f"  Min: {min(lower_shadows):.1f}%")
            print(f"  Max: {max(lower_shadows):.1f}%")
            print(f"  Mean: {np.mean(lower_shadows):.1f}%")
            print(f"  Median: {np.median(lower_shadows):.1f}%")

            print(f"\nBody Position:")
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

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("\nBased on this analysis, we can understand:")
    print("1. What exact thresholds TA-Lib uses for Shooting Star")
    print("2. Whether TA-Lib checks trend context automatically")
    print("3. How we should fix our shooting_star.py detector")


if __name__ == "__main__":
    main()

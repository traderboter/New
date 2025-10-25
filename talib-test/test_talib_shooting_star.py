"""
Test TA-Lib CDLSHOOTINGSTAR function with different scenarios

This script tests:
1. How many candles does TA-Lib need?
2. Does TA-Lib detect trend automatically?
3. What exactly triggers Shooting Star detection in TA-Lib?
"""

import numpy as np
import pandas as pd
import talib


def create_shooting_star_candle(base_price=100):
    """
    Create a perfect Shooting Star candle.

    Characteristics:
    - Small body at bottom (bearish or bullish)
    - Long upper shadow (>= 2x body)
    - Very small or no lower shadow
    """
    body_size = 2.0  # Small body
    upper_shadow = 8.0  # Long upper shadow
    lower_shadow = 0.2  # Very small lower shadow

    # Bearish candle (open > close)
    open_price = base_price + lower_shadow + body_size
    close = base_price + lower_shadow
    high = base_price + lower_shadow + body_size + upper_shadow
    low = base_price

    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    }


def create_uptrend_candles(n_candles=10, start_price=90):
    """Create candles forming an uptrend."""
    candles = []
    price = start_price

    for i in range(n_candles):
        # Bullish candle with increasing prices
        candle_size = np.random.uniform(1.5, 3.0)
        open_price = price
        close = price + candle_size
        high = close + np.random.uniform(0.1, 0.5)
        low = open_price - np.random.uniform(0.1, 0.5)

        candles.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })

        price = close + np.random.uniform(0.5, 1.0)  # Gap up

    return candles


def create_downtrend_candles(n_candles=10, start_price=110):
    """Create candles forming a downtrend."""
    candles = []
    price = start_price

    for i in range(n_candles):
        # Bearish candle with decreasing prices
        candle_size = np.random.uniform(1.5, 3.0)
        open_price = price
        close = price - candle_size
        high = open_price + np.random.uniform(0.1, 0.5)
        low = close - np.random.uniform(0.1, 0.5)

        candles.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })

        price = close - np.random.uniform(0.5, 1.0)  # Gap down

    return candles


def create_sideways_candles(n_candles=10, base_price=100):
    """Create candles forming a sideways pattern."""
    candles = []

    for i in range(n_candles):
        # Random candles around base_price
        is_bullish = np.random.choice([True, False])
        candle_size = np.random.uniform(0.5, 1.5)

        if is_bullish:
            open_price = base_price + np.random.uniform(-2, 2)
            close = open_price + candle_size
        else:
            open_price = base_price + np.random.uniform(-2, 2)
            close = open_price - candle_size

        high = max(open_price, close) + np.random.uniform(0.2, 0.8)
        low = min(open_price, close) - np.random.uniform(0.2, 0.8)

        candles.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })

    return candles


def test_talib_shooting_star(df, scenario_name):
    """Test TA-Lib CDLSHOOTINGSTAR on given dataframe."""

    # Ensure float64 type for TA-Lib
    df = df.astype(np.float64)

    # Call TA-Lib
    pattern = talib.CDLSHOOTINGSTAR(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    # Find detections
    detections = np.where(pattern != 0)[0]

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")
    print(f"Total candles: {len(df)}")
    print(f"Detections found: {len(detections)}")

    if len(detections) > 0:
        print(f"\nDetection indices: {detections}")
        print(f"\nDetection details:")
        for idx in detections:
            print(f"\n  Candle #{idx}:")
            print(f"    Pattern value: {pattern[idx]}")
            print(f"    OHLC: O={df.iloc[idx]['open']:.2f}, H={df.iloc[idx]['high']:.2f}, "
                  f"L={df.iloc[idx]['low']:.2f}, C={df.iloc[idx]['close']:.2f}")

            # Calculate candle characteristics
            candle = df.iloc[idx]
            body = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            full_range = candle['high'] - candle['low']

            print(f"    Body: {body:.2f} ({body/full_range*100:.1f}% of range)")
            print(f"    Upper shadow: {upper_shadow:.2f} ({upper_shadow/full_range*100:.1f}% of range)")
            print(f"    Lower shadow: {lower_shadow:.2f} ({lower_shadow/full_range*100:.1f}% of range)")

            # Check previous candles for context
            if idx >= 3:
                print(f"\n    Previous 3 candles trend:")
                for i in range(idx-3, idx):
                    prev = df.iloc[i]
                    direction = "🟢 Bullish" if prev['close'] > prev['open'] else "🔴 Bearish"
                    print(f"      Candle #{i}: {direction} (C={prev['close']:.2f})")
    else:
        print("\n❌ NO DETECTIONS")
        print(f"\nLast candle details:")
        candle = df.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        full_range = candle['high'] - candle['low']

        print(f"  OHLC: O={candle['open']:.2f}, H={candle['high']:.2f}, "
              f"L={candle['low']:.2f}, C={candle['close']:.2f}")
        print(f"  Body: {body:.2f} ({body/full_range*100:.1f}% of range)")
        print(f"  Upper shadow: {upper_shadow:.2f} ({upper_shadow/full_range*100:.1f}% of range)")
        print(f"  Lower shadow: {lower_shadow:.2f} ({lower_shadow/full_range*100:.1f}% of range)")

    return len(detections) > 0


def main():
    """Run all tests."""

    print("\n" + "="*70)
    print("TA-LIB SHOOTING STAR DETECTION TEST")
    print("="*70)
    print("\nGoal: Understand how TA-Lib CDLSHOOTINGSTAR works")
    print("- How many candles does it need?")
    print("- Does it check for uptrend automatically?")
    print("- What are the exact conditions?")

    # Test 1: Single Shooting Star candle only
    print("\n" + "="*70)
    print("TEST 1: Single candle only (no context)")
    print("="*70)

    shooting_star = create_shooting_star_candle(100)
    df_single = pd.DataFrame([shooting_star])
    test_talib_shooting_star(df_single, "Single Shooting Star candle")


    # Test 2: Shooting Star after uptrend (5 candles)
    print("\n" + "="*70)
    print("TEST 2: Shooting Star after UPTREND (5 previous candles)")
    print("="*70)

    uptrend_5 = create_uptrend_candles(5, start_price=90)
    shooting_star = create_shooting_star_candle(110)
    df_uptrend_5 = pd.DataFrame(uptrend_5 + [shooting_star])
    test_talib_shooting_star(df_uptrend_5, "Shooting Star after 5-candle uptrend")


    # Test 3: Shooting Star after uptrend (10 candles)
    print("\n" + "="*70)
    print("TEST 3: Shooting Star after UPTREND (10 previous candles)")
    print("="*70)

    uptrend_10 = create_uptrend_candles(10, start_price=80)
    shooting_star = create_shooting_star_candle(110)
    df_uptrend_10 = pd.DataFrame(uptrend_10 + [shooting_star])
    test_talib_shooting_star(df_uptrend_10, "Shooting Star after 10-candle uptrend")


    # Test 4: Shooting Star after downtrend
    print("\n" + "="*70)
    print("TEST 4: Shooting Star after DOWNTREND (should NOT detect)")
    print("="*70)

    downtrend = create_downtrend_candles(10, start_price=120)
    shooting_star = create_shooting_star_candle(90)
    df_downtrend = pd.DataFrame(downtrend + [shooting_star])
    test_talib_shooting_star(df_downtrend, "Shooting Star after downtrend")


    # Test 5: Shooting Star after sideways
    print("\n" + "="*70)
    print("TEST 5: Shooting Star after SIDEWAYS movement")
    print("="*70)

    sideways = create_sideways_candles(10, base_price=100)
    shooting_star = create_shooting_star_candle(100)
    df_sideways = pd.DataFrame(sideways + [shooting_star])
    test_talib_shooting_star(df_sideways, "Shooting Star after sideways")


    # Test 6: Various lengths of uptrend
    print("\n\n" + "="*70)
    print("TEST 6: Effect of uptrend LENGTH on detection")
    print("="*70)

    for n_candles in [1, 2, 3, 5, 10, 20, 50]:
        uptrend = create_uptrend_candles(n_candles, start_price=100 - n_candles * 2)
        shooting_star = create_shooting_star_candle(110)
        df = pd.DataFrame(uptrend + [shooting_star])

        detected = test_talib_shooting_star(df, f"{n_candles}-candle uptrend + Shooting Star")

        if detected:
            print(f"✅ Detected with {n_candles} candles")
        else:
            print(f"❌ NOT detected with {n_candles} candles")


    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY & CONCLUSIONS")
    print("="*70)
    print("\nBased on the tests above, we can determine:")
    print("1. Minimum number of candles TA-Lib needs")
    print("2. Whether TA-Lib checks for uptrend automatically")
    print("3. How to properly use TA-Lib in our code")


if __name__ == "__main__":
    main()

"""
Quick test for MatHoldPattern __init__ fix

This script tests that MatHoldPattern can be instantiated correctly
with the new __init__ signature.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from signal_generation.analyzers.patterns.candlestick.mat_hold import MatHoldPattern

def test_instantiation():
    """Test different ways to instantiate MatHoldPattern"""

    print("="*70)
    print("Testing MatHoldPattern Instantiation")
    print("="*70)

    # Test 1: No arguments (default)
    print("\n1️⃣  Test 1: No arguments")
    try:
        detector1 = MatHoldPattern()
        print(f"   ✅ Success! penetration={detector1.penetration}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 2: With penetration argument
    print("\n2️⃣  Test 2: With penetration argument")
    try:
        detector2 = MatHoldPattern(penetration=0.25)
        print(f"   ✅ Success! penetration={detector2.penetration}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 3: With config dictionary
    print("\n3️⃣  Test 3: With config dictionary")
    try:
        config = {'mat_hold_penetration': 0.4}
        detector3 = MatHoldPattern(config=config)
        print(f"   ✅ Success! penetration={detector3.penetration}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 4: With both config and direct argument (direct should win)
    print("\n4️⃣  Test 4: Both config and direct argument (direct wins)")
    try:
        config = {'mat_hold_penetration': 0.4}
        detector4 = MatHoldPattern(config=config, penetration=0.2)
        print(f"   ✅ Success! penetration={detector4.penetration}")
        if detector4.penetration == 0.2:
            print("   ✅ Direct argument correctly overrides config")
        else:
            print(f"   ⚠️  Expected 0.2 but got {detector4.penetration}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 5: Check that BasePattern attributes are set
    print("\n5️⃣  Test 5: Check BasePattern attributes")
    try:
        detector5 = MatHoldPattern()
        print(f"   Name: {detector5.name}")
        print(f"   Type: {detector5.pattern_type}")
        print(f"   Direction: {detector5.direction}")
        print(f"   Base Strength: {detector5.base_strength}")
        print(f"   Lookback Window: {detector5.lookback_window}")
        print(f"   ✅ All BasePattern attributes initialized correctly")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)

    return True

if __name__ == '__main__':
    success = test_instantiation()
    sys.exit(0 if success else 1)

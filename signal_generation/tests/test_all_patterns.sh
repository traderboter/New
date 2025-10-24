#!/bin/bash
#
# تست تمام الگوها به صورت خودکار
#
# استفاده:
#   chmod +x test_all_patterns.sh
#   ./test_all_patterns.sh
#

echo "=================================="
echo "🎯 Testing All Patterns"
echo "=================================="
echo ""

DATA_DIR="historical/BTC-USDT"

# رنگ‌ها برای خروجی
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# تابع تست
test_pattern() {
    pattern_name=$1
    echo ""
    echo "=================================="
    echo -e "${YELLOW}Testing: $pattern_name${NC}"
    echo "=================================="

    if python signal_generation/tests/test_pattern.py --pattern "$pattern_name" --data-dir "$DATA_DIR"; then
        echo -e "${GREEN}✓ $pattern_name: PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ $pattern_name: FAILED${NC}"
        return 1
    fi
}

# آرایه برای ذخیره نتایج
declare -a passed_patterns
declare -a failed_patterns

# Candlestick Patterns - Bullish
echo ""
echo "🟢 Bullish Candlestick Patterns"
echo "=================================="

test_pattern "hammer" && passed_patterns+=("hammer") || failed_patterns+=("hammer")
test_pattern "inverted hammer" && passed_patterns+=("inverted hammer") || failed_patterns+=("inverted hammer")
test_pattern "engulfing" && passed_patterns+=("engulfing") || failed_patterns+=("engulfing")
test_pattern "morning star" && passed_patterns+=("morning star") || failed_patterns+=("morning star")
test_pattern "piercing line" && passed_patterns+=("piercing line") || failed_patterns+=("piercing line")
test_pattern "three white soldiers" && passed_patterns+=("three white soldiers") || failed_patterns+=("three white soldiers")
test_pattern "morning doji star" && passed_patterns+=("morning doji star") || failed_patterns+=("morning doji star")

# Candlestick Patterns - Bearish
echo ""
echo "🔴 Bearish Candlestick Patterns"
echo "=================================="

test_pattern "shooting star" && passed_patterns+=("shooting star") || failed_patterns+=("shooting star")
test_pattern "hanging man" && passed_patterns+=("hanging man") || failed_patterns+=("hanging man")
test_pattern "evening star" && passed_patterns+=("evening star") || failed_patterns+=("evening star")
test_pattern "dark cloud cover" && passed_patterns+=("dark cloud cover") || failed_patterns+=("dark cloud cover")
test_pattern "three black crows" && passed_patterns+=("three black crows") || failed_patterns+=("three black crows")
test_pattern "evening doji star" && passed_patterns+=("evening doji star") || failed_patterns+=("evening doji star")

# Candlestick Patterns - Reversal
echo ""
echo "🔄 Reversal Candlestick Patterns"
echo "=================================="

test_pattern "doji" && passed_patterns+=("doji") || failed_patterns+=("doji")
test_pattern "harami" && passed_patterns+=("harami") || failed_patterns+=("harami")
test_pattern "harami cross" && passed_patterns+=("harami cross") || failed_patterns+=("harami cross")

# Chart Patterns
echo ""
echo "📊 Chart Patterns"
echo "=================================="

test_pattern "double top bottom" && passed_patterns+=("double top bottom") || failed_patterns+=("double top bottom")
test_pattern "head shoulders" && passed_patterns+=("head shoulders") || failed_patterns+=("head shoulders")
test_pattern "triangle" && passed_patterns+=("triangle") || failed_patterns+=("triangle")
test_pattern "wedge" && passed_patterns+=("wedge") || failed_patterns+=("wedge")

# خلاصه نتایج
echo ""
echo "=================================="
echo "📋 Test Summary"
echo "=================================="
echo ""

passed_count=${#passed_patterns[@]}
failed_count=${#failed_patterns[@]}
total_count=$((passed_count + failed_count))

echo -e "${GREEN}Passed: $passed_count / $total_count${NC}"
if [ $passed_count -gt 0 ]; then
    for pattern in "${passed_patterns[@]}"; do
        echo -e "  ${GREEN}✓${NC} $pattern"
    done
fi

echo ""
echo -e "${RED}Failed: $failed_count / $total_count${NC}"
if [ $failed_count -gt 0 ]; then
    for pattern in "${failed_patterns[@]}"; do
        echo -e "  ${RED}✗${NC} $pattern"
    done
fi

echo ""
echo "=================================="
if [ $failed_count -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed${NC}"
    exit 1
fi

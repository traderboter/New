#!/bin/bash

# Script to run indicator tests with various options

echo "================================================"
echo "  Indicators Test Suite Runner"
echo "================================================"
echo ""

# Set Python path
export PYTHONPATH=/home/user/New:$PYTHONPATH

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
TEST_TYPE=${1:-all}

case $TEST_TYPE in
    all)
        echo -e "${BLUE}Running all tests...${NC}"
        pytest Indicators_Test/ -v
        ;;

    fast)
        echo -e "${BLUE}Running tests in parallel (fast mode)...${NC}"
        pytest Indicators_Test/ -n auto
        ;;

    coverage)
        echo -e "${BLUE}Running tests with coverage report...${NC}"
        pytest Indicators_Test/ \
            --cov=signal_generation.analyzers.indicators \
            --cov-report=html \
            --cov-report=term-missing \
            -v
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;

    specific)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Usage: $0 specific <test_file_or_pattern>${NC}"
            echo "Example: $0 specific test_rsi.py"
            exit 1
        fi
        echo -e "${BLUE}Running specific test: $2${NC}"
        pytest Indicators_Test/$2 -v
        ;;

    base)
        echo -e "${BLUE}Running base indicator tests...${NC}"
        pytest Indicators_Test/test_base_indicator.py -v
        ;;

    sma)
        echo -e "${BLUE}Running SMA tests...${NC}"
        pytest Indicators_Test/test_sma.py -v
        ;;

    ema)
        echo -e "${BLUE}Running EMA tests...${NC}"
        pytest Indicators_Test/test_ema.py -v
        ;;

    rsi)
        echo -e "${BLUE}Running RSI tests...${NC}"
        pytest Indicators_Test/test_rsi.py -v
        ;;

    macd)
        echo -e "${BLUE}Running MACD tests...${NC}"
        pytest Indicators_Test/test_macd.py -v
        ;;

    stochastic)
        echo -e "${BLUE}Running Stochastic tests...${NC}"
        pytest Indicators_Test/test_stochastic.py -v
        ;;

    atr)
        echo -e "${BLUE}Running ATR tests...${NC}"
        pytest Indicators_Test/test_atr.py -v
        ;;

    bollinger)
        echo -e "${BLUE}Running Bollinger Bands tests...${NC}"
        pytest Indicators_Test/test_bollinger_bands.py -v
        ;;

    obv)
        echo -e "${BLUE}Running OBV tests...${NC}"
        pytest Indicators_Test/test_obv.py -v
        ;;

    orchestrator)
        echo -e "${BLUE}Running Orchestrator tests...${NC}"
        pytest Indicators_Test/test_indicator_orchestrator.py -v
        ;;

    help|--help|-h)
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  all          - Run all tests (default)"
        echo "  fast         - Run tests in parallel"
        echo "  coverage     - Run tests with coverage report"
        echo "  specific     - Run specific test file"
        echo "  base         - Run base indicator tests"
        echo "  sma          - Run SMA tests"
        echo "  ema          - Run EMA tests"
        echo "  rsi          - Run RSI tests"
        echo "  macd         - Run MACD tests"
        echo "  stochastic   - Run Stochastic tests"
        echo "  atr          - Run ATR tests"
        echo "  bollinger    - Run Bollinger Bands tests"
        echo "  obv          - Run OBV tests"
        echo "  orchestrator - Run Orchestrator tests"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 all"
        echo "  $0 fast"
        echo "  $0 coverage"
        echo "  $0 specific test_rsi.py"
        exit 0
        ;;

    *)
        echo -e "${YELLOW}Unknown option: $TEST_TYPE${NC}"
        echo "Use '$0 help' to see available options"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Test execution completed!${NC}"

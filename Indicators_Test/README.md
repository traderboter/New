# 📊 Indicators Test Suite

Comprehensive test suite for all technical indicators in the trading bot.

## 📁 Test Files

| Test File | Description | Coverage |
|-----------|-------------|----------|
| `test_base_indicator.py` | Tests for abstract base class | Initialization, validation, caching, error handling |
| `test_sma.py` | Simple Moving Average tests | Calculation, multiple periods, edge cases |
| `test_ema.py` | Exponential Moving Average tests | EMA formula, responsiveness, Wilder's smoothing |
| `test_rsi.py` | Relative Strength Index tests | Range validation, trend detection, Wilder's smoothing |
| `test_macd.py` | MACD tests | Component relationships, crossovers, trends |
| `test_stochastic.py` | Stochastic Oscillator tests | %K and %D, overbought/oversold conditions |
| `test_atr.py` | Average True Range tests | Volatility measurement, True Range calculation |
| `test_bollinger_bands.py` | Bollinger Bands tests | Band relationships, volatility changes |
| `test_obv.py` | On-Balance Volume tests | Volume accumulation, price-volume relationship |
| `test_indicator_orchestrator.py` | Orchestrator tests | Multi-indicator calculation, coordination |

## 🧪 Test Categories

### 1. **Functionality Tests**
- ✅ Correct calculation formulas
- ✅ Output column generation
- ✅ Value ranges and constraints

### 2. **Edge Case Tests**
- ✅ Flat prices (no movement)
- ✅ Extreme volatility
- ✅ Zero/negative/NaN values
- ✅ Insufficient data
- ✅ Very short/long periods

### 3. **Integration Tests**
- ✅ Multiple indicators working together
- ✅ Dependency management
- ✅ Performance with large datasets

### 4. **Error Handling Tests**
- ✅ Missing columns
- ✅ Empty DataFrames
- ✅ Invalid configurations
- ✅ Calculation errors

## 🚀 Running Tests

### Run All Tests
```bash
cd /home/user/New
pytest Indicators_Test/ -v
```

### Run Specific Test File
```bash
pytest Indicators_Test/test_rsi.py -v
```

### Run with Coverage
```bash
pytest Indicators_Test/ --cov=signal_generation.analyzers.indicators --cov-report=html
```

### Run Specific Test
```bash
pytest Indicators_Test/test_rsi.py::TestRSIIndicator::test_rsi_range -v
```

### Run Tests in Parallel (faster)
```bash
pytest Indicators_Test/ -n auto
```

## 📋 Test Fixtures

Located in `conftest.py`:

- **sample_ohlcv_data**: 100 rows of realistic OHLCV data
- **small_ohlcv_data**: 10 rows for edge case testing
- **flat_price_data**: 50 rows of flat price (no movement)
- **empty_dataframe**: Empty DataFrame for error testing
- **config_default**: Default configuration
- **config_custom**: Custom configuration for parameter testing

## 📊 Expected Test Results

All tests should pass. If you see failures, check:

1. **Import Errors**: Ensure all dependencies are installed
2. **Path Issues**: Run tests from the correct directory
3. **Data Issues**: Check if test fixtures are generating correct data

## 🔍 Test Statistics

- **Total Test Files**: 10
- **Estimated Test Count**: ~150+ tests
- **Coverage Target**: >90%

## 📝 Writing New Tests

When adding new indicators:

1. Create `test_<indicator_name>.py`
2. Include these test categories:
   - Basic calculation
   - Formula correctness
   - Edge cases
   - Error handling
   - Integration
3. Use fixtures from `conftest.py`
4. Follow naming convention: `test_<functionality>`

### Example Test Template

```python
import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.your_indicator import YourIndicator


class TestYourIndicator:
    def test_initialization_default(self):
        indicator = YourIndicator()
        assert indicator.name == "YourIndicator"
        # Add more assertions

    def test_calculate_basic(self, sample_ohlcv_data):
        indicator = YourIndicator()
        result_df = indicator.calculate_safe(sample_ohlcv_data)
        assert 'your_column' in result_df.columns
        # Add more assertions

    def test_edge_case_flat_price(self, flat_price_data):
        indicator = YourIndicator()
        result_df = indicator.calculate_safe(flat_price_data)
        # Test behavior with flat price
```

## 🐛 Common Issues

### Issue: ModuleNotFoundError
**Solution**: Ensure you're in the correct directory and Python path is set:
```bash
export PYTHONPATH=/home/user/New:$PYTHONPATH
```

### Issue: Fixture not found
**Solution**: Make sure `conftest.py` is in the test directory

### Issue: Tests are slow
**Solution**: Run with parallel execution:
```bash
pytest Indicators_Test/ -n auto
```

## 📈 Code Quality Metrics

Tests validate:
- ✅ **Correctness**: Calculations match expected formulas
- ✅ **Robustness**: Handles edge cases gracefully
- ✅ **Performance**: Efficient with large datasets
- ✅ **Maintainability**: Clear, documented test code

## 🎯 Continuous Integration

These tests are designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Indicator Tests
  run: |
    pytest Indicators_Test/ -v --cov
```

## 📞 Support

If tests fail or you need help:
1. Check the error message carefully
2. Review the specific test file
3. Ensure all dependencies are installed
4. Check indicator implementation against test expectations

---

**Last Updated**: 2025-01-27
**Total Test Coverage**: Comprehensive coverage of all 8 indicators + base class + orchestrator

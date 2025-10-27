"""
Tests for Indicator Orchestrator.
"""

import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.indicator_orchestrator import IndicatorOrchestrator
from signal_generation.analyzers.indicators.base_indicator import BaseIndicator


class TestIndicatorOrchestrator:
    """Test suite for IndicatorOrchestrator."""

    def test_initialization(self, config_default):
        """Test orchestrator initialization."""
        orchestrator = IndicatorOrchestrator(config_default)

        assert orchestrator is not None
        assert len(orchestrator.all_indicators) > 0

    def test_load_all_indicators(self, config_default):
        """Test that all indicators are loaded."""
        orchestrator = IndicatorOrchestrator(config_default)

        # Should have all indicator types
        assert len(orchestrator.trend_indicators) > 0
        assert len(orchestrator.momentum_indicators) > 0
        assert len(orchestrator.volatility_indicators) > 0
        assert len(orchestrator.volume_indicators) > 0

        # Check specific indicators
        assert 'EMA' in orchestrator.all_indicators
        assert 'SMA' in orchestrator.all_indicators
        assert 'RSI' in orchestrator.all_indicators
        assert 'MACD' in orchestrator.all_indicators
        assert 'ATR' in orchestrator.all_indicators
        assert 'Bollinger Bands' in orchestrator.all_indicators
        assert 'Stochastic' in orchestrator.all_indicators
        assert 'OBV' in orchestrator.all_indicators

    def test_get_indicator(self, config_default):
        """Test getting specific indicator."""
        orchestrator = IndicatorOrchestrator(config_default)

        ema = orchestrator.get_indicator('EMA')
        assert ema is not None
        assert ema.name == 'EMA'

        rsi = orchestrator.get_indicator('RSI')
        assert rsi is not None
        assert rsi.name == 'RSI'

        # Non-existent indicator
        none_indicator = orchestrator.get_indicator('NonExistent')
        assert none_indicator is None

    def test_get_available_indicators(self, config_default):
        """Test getting list of available indicators."""
        orchestrator = IndicatorOrchestrator(config_default)

        available = orchestrator.get_available_indicators()

        assert isinstance(available, dict)
        assert 'trend' in available
        assert 'momentum' in available
        assert 'volatility' in available
        assert 'volume' in available

        # Check trend indicators
        assert 'EMA' in available['trend']
        assert 'SMA' in available['trend']

        # Check momentum indicators
        assert 'RSI' in available['momentum']
        assert 'MACD' in available['momentum']

    def test_calculate_all_indicators(self, sample_ohlcv_data, config_default):
        """Test calculating all indicators."""
        orchestrator = IndicatorOrchestrator(config_default)

        result_df = orchestrator.calculate_all(sample_ohlcv_data)

        # Should have all indicator columns
        assert 'ema_20' in result_df.columns or 'ema_50' in result_df.columns
        assert 'sma_20' in result_df.columns or 'sma_50' in result_df.columns
        assert 'rsi' in result_df.columns
        assert 'macd' in result_df.columns
        assert 'atr' in result_df.columns
        assert 'bb_upper' in result_df.columns
        assert 'stoch_k' in result_df.columns
        assert 'obv' in result_df.columns

        # Original columns should still be there
        assert 'close' in result_df.columns
        assert 'high' in result_df.columns
        assert 'low' in result_df.columns

    def test_calculate_specific_indicators(self, sample_ohlcv_data, config_default):
        """Test calculating specific indicators."""
        orchestrator = IndicatorOrchestrator(config_default)

        result_df = orchestrator.calculate_all(
            sample_ohlcv_data,
            indicator_names=['RSI', 'MACD']
        )

        # Should have requested indicators
        assert 'rsi' in result_df.columns
        assert 'macd' in result_df.columns

        # Should not have other indicators (or they should be from original data)
        # Actually, calculate_all adds indicators, doesn't remove existing ones
        # So we just check the requested ones exist

    def test_calculate_by_type_trend(self, sample_ohlcv_data, config_default):
        """Test calculating trend indicators."""
        orchestrator = IndicatorOrchestrator(config_default)

        result_df = orchestrator.calculate_by_type(sample_ohlcv_data, 'trend')

        # Should have trend indicators
        assert 'ema_20' in result_df.columns or 'ema_50' in result_df.columns
        assert 'sma_20' in result_df.columns or 'sma_50' in result_df.columns

    def test_calculate_by_type_momentum(self, sample_ohlcv_data, config_default):
        """Test calculating momentum indicators."""
        orchestrator = IndicatorOrchestrator(config_default)

        result_df = orchestrator.calculate_by_type(sample_ohlcv_data, 'momentum')

        # Should have momentum indicators
        assert 'rsi' in result_df.columns
        assert 'macd' in result_df.columns
        assert 'stoch_k' in result_df.columns

    def test_calculate_by_type_volatility(self, sample_ohlcv_data, config_default):
        """Test calculating volatility indicators."""
        orchestrator = IndicatorOrchestrator(config_default)

        result_df = orchestrator.calculate_by_type(sample_ohlcv_data, 'volatility')

        # Should have volatility indicators
        assert 'atr' in result_df.columns
        assert 'bb_upper' in result_df.columns

    def test_calculate_by_type_volume(self, sample_ohlcv_data, config_default):
        """Test calculating volume indicators."""
        orchestrator = IndicatorOrchestrator(config_default)

        result_df = orchestrator.calculate_by_type(sample_ohlcv_data, 'volume')

        # Should have volume indicators
        assert 'obv' in result_df.columns

    def test_calculation_order(self, sample_ohlcv_data, config_default):
        """Test that indicators are calculated in correct order."""
        orchestrator = IndicatorOrchestrator(config_default)

        # The orchestrator should calculate in order:
        # trend -> momentum -> volatility -> volume
        # This test just verifies it completes successfully
        result_df = orchestrator.calculate_all(sample_ohlcv_data)

        assert len(result_df) == len(sample_ohlcv_data)

    def test_get_indicator_value(self, sample_ohlcv_data, config_default):
        """Test getting indicator value."""
        orchestrator = IndicatorOrchestrator(config_default)

        result_df = orchestrator.calculate_all(sample_ohlcv_data)

        rsi_value = orchestrator.get_indicator_value(result_df, 'RSI', 'rsi')

        assert rsi_value is not None
        assert isinstance(rsi_value, pd.Series)

    def test_statistics_tracking(self, sample_ohlcv_data, config_default):
        """Test statistics tracking."""
        orchestrator = IndicatorOrchestrator(config_default)

        # Initial stats
        initial_stats = orchestrator.get_stats()
        assert initial_stats['total_calculations'] == 0

        # Calculate indicators
        orchestrator.calculate_all(sample_ohlcv_data)

        # Stats should be updated
        stats = orchestrator.get_stats()
        assert stats['total_calculations'] > 0

        # Reset stats
        orchestrator.reset_stats()
        stats = orchestrator.get_stats()
        assert stats['total_calculations'] == 0

    def test_clear_all_caches(self, sample_ohlcv_data, config_default):
        """Test clearing all indicator caches."""
        config_default['cache_enabled'] = True
        orchestrator = IndicatorOrchestrator(config_default)

        # Calculate to populate caches
        orchestrator.calculate_all(sample_ohlcv_data)

        # Clear caches
        orchestrator.clear_all_caches()

        # Verify caches are cleared (check one indicator)
        rsi = orchestrator.get_indicator('RSI')
        assert rsi._last_hash is None
        assert rsi._last_result is None

    def test_error_handling(self, config_default):
        """Test error handling with invalid data."""
        orchestrator = IndicatorOrchestrator(config_default)

        # Empty DataFrame
        empty_df = pd.DataFrame()
        result_df = orchestrator.calculate_all(empty_df)

        # Should return empty DataFrame without crashing
        assert len(result_df) == 0

    def test_partial_data(self, small_ohlcv_data, config_default):
        """Test with partial data (some indicators may fail)."""
        orchestrator = IndicatorOrchestrator(config_default)

        result_df = orchestrator.calculate_all(small_ohlcv_data)

        # Some indicators may not calculate due to insufficient data
        # but orchestrator should handle gracefully
        assert len(result_df) == len(small_ohlcv_data)

    def test_str_representation(self, config_default):
        """Test string representation."""
        orchestrator = IndicatorOrchestrator(config_default)

        str_repr = str(orchestrator)

        assert "IndicatorOrchestrator" in str_repr
        assert "total=" in str_repr
        assert "trend=" in str_repr
        assert "momentum=" in str_repr

    def test_custom_indicator_registration(self, config_default):
        """Test registering custom indicator."""
        orchestrator = IndicatorOrchestrator(config_default)

        initial_count = len(orchestrator.all_indicators)

        # Create a simple custom indicator
        class CustomIndicator(BaseIndicator):
            def _get_indicator_name(self) -> str:
                return "Custom"

            def _get_indicator_type(self) -> str:
                return "other"

            def _get_required_columns(self):
                return ['close']

            def _get_output_columns(self):
                return ['custom_value']

            def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
                result = df.copy()
                result['custom_value'] = df['close'] * 2
                return result

        # Register it
        orchestrator.register_indicator(CustomIndicator)

        # Should be added
        assert len(orchestrator.all_indicators) == initial_count + 1
        assert 'Custom' in orchestrator.all_indicators

    def test_invalid_indicator_type(self, sample_ohlcv_data, config_default):
        """Test calculating with invalid indicator type."""
        orchestrator = IndicatorOrchestrator(config_default)

        result_df = orchestrator.calculate_by_type(sample_ohlcv_data, 'invalid_type')

        # Should return original DataFrame
        assert len(result_df) == len(sample_ohlcv_data)


class TestIndicatorOrchestratorIntegration:
    """Integration tests for IndicatorOrchestrator."""

    def test_full_pipeline(self, sample_ohlcv_data, config_default):
        """Test full calculation pipeline."""
        orchestrator = IndicatorOrchestrator(config_default)

        # Calculate all indicators
        result_df = orchestrator.calculate_all(sample_ohlcv_data)

        # Verify we have a complete set of indicators
        expected_columns = [
            'close', 'high', 'low', 'open', 'volume',  # Original
            'rsi', 'macd', 'atr', 'obv'  # Indicators (some examples)
        ]

        for col in expected_columns:
            if col in ['open', 'timestamp']:  # These might not be in sample data
                continue
            assert col in result_df.columns

        # Get latest values
        rsi_latest = orchestrator.get_indicator_value(result_df, 'RSI', 'rsi')
        assert rsi_latest is not None

    def test_performance_with_large_dataset(self, config_default):
        """Test performance with larger dataset."""
        # Create larger dataset
        np.random.seed(42)
        large_df = pd.DataFrame({
            'open': np.random.normal(50000, 1000, 1000),
            'high': np.random.normal(51000, 1000, 1000),
            'low': np.random.normal(49000, 1000, 1000),
            'close': np.random.normal(50000, 1000, 1000),
            'volume': np.random.normal(1000000, 100000, 1000)
        })

        orchestrator = IndicatorOrchestrator(config_default)

        # Should complete in reasonable time
        result_df = orchestrator.calculate_all(large_df)

        assert len(result_df) == 1000
        assert len(result_df.columns) > len(large_df.columns)

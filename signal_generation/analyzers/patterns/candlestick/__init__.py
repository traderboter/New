"""
Candlestick Patterns Module

Individual candlestick pattern detectors.
Each pattern has its own file for better maintainability.
"""

from signal_generation.analyzers.patterns.candlestick.hammer import HammerPattern
from signal_generation.analyzers.patterns.candlestick.inverted_hammer import InvertedHammerPattern
from signal_generation.analyzers.patterns.candlestick.engulfing import EngulfingPattern
from signal_generation.analyzers.patterns.candlestick.morning_star import MorningStarPattern
from signal_generation.analyzers.patterns.candlestick.piercing_line import PiercingLinePattern
from signal_generation.analyzers.patterns.candlestick.three_white_soldiers import ThreeWhiteSoldiersPattern
from signal_generation.analyzers.patterns.candlestick.morning_doji_star import MorningDojiStarPattern
from signal_generation.analyzers.patterns.candlestick.shooting_star import ShootingStarPattern
from signal_generation.analyzers.patterns.candlestick.hanging_man import HangingManPattern
from signal_generation.analyzers.patterns.candlestick.evening_star import EveningStarPattern
from signal_generation.analyzers.patterns.candlestick.dark_cloud_cover import DarkCloudCoverPattern
from signal_generation.analyzers.patterns.candlestick.three_black_crows import ThreeBlackCrowsPattern
from signal_generation.analyzers.patterns.candlestick.evening_doji_star import EveningDojiStarPattern
from signal_generation.analyzers.patterns.candlestick.doji import DojiPattern
from signal_generation.analyzers.patterns.candlestick.harami import HaramiPattern
from signal_generation.analyzers.patterns.candlestick.harami_cross import HaramiCrossPattern

__all__ = [
    # Bullish patterns
    'HammerPattern',
    'InvertedHammerPattern',
    'MorningStarPattern',
    'PiercingLinePattern',
    'ThreeWhiteSoldiersPattern',
    'MorningDojiStarPattern',

    # Bearish patterns
    'ShootingStarPattern',
    'HangingManPattern',
    'EveningStarPattern',
    'DarkCloudCoverPattern',
    'ThreeBlackCrowsPattern',
    'EveningDojiStarPattern',

    # Reversal/Both
    'EngulfingPattern',
    'DojiPattern',
    'HaramiPattern',
    'HaramiCrossPattern',
]

"""
Chart Patterns Module

Individual chart pattern detectors.
Each pattern has its own file for better maintainability.
"""

from signal_generation.analyzers.patterns.chart.double_top_bottom import DoubleTopBottomPattern
from signal_generation.analyzers.patterns.chart.head_shoulders import HeadShouldersPattern
from signal_generation.analyzers.patterns.chart.triangle import TrianglePattern
from signal_generation.analyzers.patterns.chart.wedge import WedgePattern

__all__ = [
    'DoubleTopBottomPattern',
    'HeadShouldersPattern',
    'TrianglePattern',
    'WedgePattern',
]

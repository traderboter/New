"""
SignalScore - Signal Scoring Data Structure

Holds comprehensive scoring information for a trading signal.
Includes base score, weighted components, bonuses, and final score.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignalScore:
    """
    Comprehensive signal scoring structure.
    
    Contains all scoring components:
    - Base scores from each analyzer
    - Weighted scores
    - Bonuses (confluence, HTF alignment)
    - Multipliers (timeframe, volatility)
    - Final score
    """
    
    # Base scores (0-100 each)
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    pattern_score: float = 0.0
    sr_score: float = 0.0
    volatility_score: float = 0.0
    harmonic_score: float = 0.0
    channel_score: float = 0.0
    cyclical_score: float = 0.0
    htf_score: float = 0.0
    
    # Weighted scores (after applying weights)
    weighted_trend: float = 0.0
    weighted_momentum: float = 0.0
    weighted_volume: float = 0.0
    weighted_pattern: float = 0.0
    weighted_sr: float = 0.0
    weighted_volatility: float = 0.0
    weighted_harmonic: float = 0.0
    weighted_channel: float = 0.0
    weighted_cyclical: float = 0.0
    weighted_htf: float = 0.0
    
    # Aggregate scores
    base_score: float = 0.0              # Sum of weighted scores
    
    # Bonuses and multipliers
    confluence_bonus: float = 0.0        # 0-0.5 (همگرایی)
    timeframe_weight: float = 1.0        # 0.5-1.5 (وزن تایم‌فریم)
    htf_multiplier: float = 1.0          # 0.7-1.3 (هم‌راستایی با HTF)
    volatility_multiplier: float = 1.0   # 0.6-1.5 (تنظیم نوسان)
    
    # Final score
    final_score: float = 0.0             # امتیاز نهایی (0-300)
    
    # Metadata
    confidence: float = 0.5              # اطمینان (0-1)
    signal_strength: str = 'weak'        # 'weak', 'medium', 'strong'
    contributing_analyzers: List[str] = field(default_factory=list)
    aligned_analyzers: int = 0
    
    # Scoring breakdown for debugging
    breakdown: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_final_score(self) -> float:
        """
        Calculate final score from all components.
        
        Formula:
        final_score = base_score 
                     * (1 + confluence_bonus)
                     * timeframe_weight
                     * htf_multiplier
                     * volatility_multiplier
        
        Returns:
            Final score (0-300)
        """
        self.final_score = (
            self.base_score
            * (1.0 + self.confluence_bonus)
            * self.timeframe_weight
            * self.htf_multiplier
            * self.volatility_multiplier
        )
        
        # Clamp between 0 and 300
        self.final_score = max(0.0, min(self.final_score, 300.0))
        
        return self.final_score
    
    def determine_signal_strength(self) -> str:
        """
        Determine signal strength category based on final score.
        
        Categories:
        - weak: < 80
        - medium: 80-150
        - strong: > 150
        
        Returns:
            Signal strength category
        """
        if self.final_score < 80:
            self.signal_strength = 'weak'
        elif self.final_score < 150:
            self.signal_strength = 'medium'
        else:
            self.signal_strength = 'strong'
        
        return self.signal_strength
    
    def calculate_confidence(self) -> float:
        """
        Calculate confidence score (0-1).
        
        Based on:
        - Number of aligned analyzers
        - Final score magnitude
        - Confluence bonus
        
        Returns:
            Confidence (0-1)
        """
        confidence = 0.3  # Base confidence
        
        # Aligned analyzers boost confidence
        if self.aligned_analyzers >= 7:
            confidence += 0.3
        elif self.aligned_analyzers >= 5:
            confidence += 0.2
        elif self.aligned_analyzers >= 3:
            confidence += 0.1
        
        # High final score boosts confidence
        if self.final_score > 200:
            confidence += 0.2
        elif self.final_score > 150:
            confidence += 0.15
        elif self.final_score > 100:
            confidence += 0.1
        
        # Confluence bonus
        confidence += self.confluence_bonus * 0.3
        
        # Clamp between 0 and 1
        self.confidence = max(0.0, min(confidence, 1.0))
        
        return self.confidence
    
    def build_breakdown(self) -> Dict[str, Any]:
        """
        Build detailed breakdown of scoring.
        
        Returns:
            Dictionary with scoring details
        """
        self.breakdown = {
            'base_scores': {
                'trend': round(self.trend_score, 2),
                'momentum': round(self.momentum_score, 2),
                'volume': round(self.volume_score, 2),
                'pattern': round(self.pattern_score, 2),
                'sr': round(self.sr_score, 2),
                'volatility': round(self.volatility_score, 2),
                'harmonic': round(self.harmonic_score, 2),
                'channel': round(self.channel_score, 2),
                'cyclical': round(self.cyclical_score, 2),
                'htf': round(self.htf_score, 2)
            },
            'weighted_scores': {
                'trend': round(self.weighted_trend, 2),
                'momentum': round(self.weighted_momentum, 2),
                'volume': round(self.weighted_volume, 2),
                'pattern': round(self.weighted_pattern, 2),
                'sr': round(self.weighted_sr, 2),
                'volatility': round(self.weighted_volatility, 2),
                'harmonic': round(self.weighted_harmonic, 2),
                'channel': round(self.weighted_channel, 2),
                'cyclical': round(self.weighted_cyclical, 2),
                'htf': round(self.weighted_htf, 2)
            },
            'aggregates': {
                'base_score': round(self.base_score, 2),
                'confluence_bonus': round(self.confluence_bonus, 3),
                'timeframe_weight': round(self.timeframe_weight, 2),
                'htf_multiplier': round(self.htf_multiplier, 2),
                'volatility_multiplier': round(self.volatility_multiplier, 2)
            },
            'final': {
                'score': round(self.final_score, 2),
                'confidence': round(self.confidence, 2),
                'strength': self.signal_strength
            },
            'meta': {
                'contributing_analyzers': self.contributing_analyzers,
                'aligned_analyzers': self.aligned_analyzers
            }
        }
        
        return self.breakdown
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"SignalScore(final={self.final_score:.2f}, "
            f"confidence={self.confidence:.2f}, "
            f"strength={self.signal_strength}, "
            f"aligned={self.aligned_analyzers})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()

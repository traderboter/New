"""
BaseAnalyzer - Abstract Base Class for All Analyzers

All analyzers must inherit from this base class and implement the analyze() method.

Standard pattern:
1. Check if enabled
2. Validate context
3. Perform analysis
4. Store results in context
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.
    
    All analyzers must:
    1. Inherit from this class
    2. Implement the analyze() method
    3. Store results in context using context.add_result()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = True
        
        logger.debug(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def analyze(self, context: 'AnalysisContext') -> None:
        """
        Main analysis method - must be implemented by all analyzers.
        
        Args:
            context: AnalysisContext with dataframe and indicators
            
        This method should:
        1. Check if enabled (using _check_enabled())
        2. Validate context (using _validate_context())
        3. Perform analysis
        4. Store results (using context.add_result())
        """
        pass
    
    def _check_enabled(self) -> bool:
        """
        Check if this analyzer is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return getattr(self, 'enabled', True)
    
    def _validate_context(self, context: 'AnalysisContext') -> bool:
        """
        Validate that context has required data.
        
        Override this method in subclasses to add specific validation.
        
        Args:
            context: AnalysisContext to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation
        if context.df is None or len(context.df) == 0:
            logger.warning(f"{self.__class__.__name__}: Empty DataFrame")
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(enabled={self.enabled})"

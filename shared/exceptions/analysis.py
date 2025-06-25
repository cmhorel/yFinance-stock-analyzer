"""Analysis-related exceptions."""
from .base import StockAnalyzerError


class AnalysisError(StockAnalyzerError):
    """Base exception for analysis-related errors."""
    pass


class InsufficientDataError(AnalysisError):
    """Exception raised when there is insufficient data for analysis."""
    pass

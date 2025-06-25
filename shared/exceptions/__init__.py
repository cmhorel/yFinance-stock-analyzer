"""Custom exceptions for the stock analyzer application."""
from .base import StockAnalyzerError
from .data import DataError, DataValidationError, DataNotFoundError
from .analysis import AnalysisError, InsufficientDataError
from .scraping import ScrapingError, RateLimitError, NetworkError
from .config import ConfigurationError

__all__ = [
    'StockAnalyzerError',
    'DataError',
    'DataValidationError', 
    'DataNotFoundError',
    'AnalysisError',
    'InsufficientDataError',
    'ScrapingError',
    'RateLimitError',
    'NetworkError',
    'ConfigurationError'
]

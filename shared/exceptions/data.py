"""Data-related exceptions."""
from .base import StockAnalyzerError


class DataError(StockAnalyzerError):
    """Base exception for data-related errors."""
    pass


class DataValidationError(DataError):
    """Exception raised when data validation fails."""
    pass


class DataNotFoundError(DataError):
    """Exception raised when requested data is not found."""
    pass

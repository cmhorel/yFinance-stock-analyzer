"""Configuration-related exceptions."""
from .base import StockAnalyzerError


class ConfigurationError(StockAnalyzerError):
    """Exception raised for configuration-related errors."""
    pass

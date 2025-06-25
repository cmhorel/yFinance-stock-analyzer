"""Scraping-related exceptions."""
from .base import StockAnalyzerError


class ScrapingError(StockAnalyzerError):
    """Base exception for scraping-related errors."""
    pass


class RateLimitError(ScrapingError):
    """Exception raised when API rate limits are exceeded."""
    pass


class NetworkError(ScrapingError):
    """Exception raised for network-related errors."""
    pass

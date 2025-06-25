"""Base exception classes for the stock analyzer application."""
from typing import Dict, Any, Optional


class StockAnalyzerError(Exception):
    """Base exception for all stock analyzer errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception with message and optional details."""
        super().__init__(message)
        self.message = message
        self.details = details if details is not None else {}
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

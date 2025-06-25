"""Logging configuration and utilities."""
import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get a logger instance for this class."""
        return get_logger(self.__class__.__name__)


def setup_logging(
    level: int = logging.INFO,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level
        format_string: Log message format
        log_file: Optional log file path
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('plotly').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically module or class name)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ContextualLogger:
    """Logger wrapper that adds contextual information to log messages."""
    
    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        """
        Initialize contextual logger.
        
        Args:
            logger: Base logger instance
            context: Dictionary of contextual information
        """
        self._logger = logger
        self._context = context if context is not None else {}
    
    def _format_message(self, message: str) -> str:
        """Format message with context information."""
        if not self._context:
            return message
        
        context_str = " | ".join(f"{k}={v}" for k, v in self._context.items())
        return f"[{context_str}] {message}"
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with context."""
        self._logger.debug(self._format_message(message), *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with context."""
        self._logger.info(self._format_message(message), *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with context."""
        self._logger.warning(self._format_message(message), *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with context."""
        self._logger.error(self._format_message(message), *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message with context."""
        self._logger.critical(self._format_message(message), *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception message with context."""
        self._logger.exception(self._format_message(message), *args, **kwargs)
    
    def add_context(self, **kwargs) -> 'ContextualLogger':
        """Add context information and return new logger instance."""
        new_context = {**self._context, **kwargs}
        return ContextualLogger(self._logger, new_context)


def get_contextual_logger(name: str, **context) -> ContextualLogger:
    """
    Get a contextual logger instance.
    
    Args:
        name: Logger name
        **context: Context information to include in log messages
        
    Returns:
        ContextualLogger instance
    """
    logger = get_logger(name)
    return ContextualLogger(logger, context)


class TimedLogger:
    """Logger that tracks execution time for operations."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        """
        Initialize timed logger.
        
        Args:
            logger: Logger instance
            operation: Description of the operation being timed
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        """Start timing the operation."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log the duration."""
        if self.start_time:
            duration = datetime.now() - self.start_time
            if exc_type:
                self.logger.error(f"Failed {self.operation} after {duration.total_seconds():.2f}s")
            else:
                self.logger.info(f"Completed {self.operation} in {duration.total_seconds():.2f}s")


def timed_operation(logger: logging.Logger, operation: str) -> TimedLogger:
    """
    Create a timed logger context manager.
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        
    Returns:
        TimedLogger context manager
    """
    return TimedLogger(logger, operation)

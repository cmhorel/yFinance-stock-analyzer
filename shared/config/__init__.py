"""Configuration management module."""
from .config import Config, get_config
from .settings import Settings

# Convenience functions
def get_settings() -> Settings:
    """Get the current settings instance."""
    return get_config().settings

def setup_logging(**kwargs):
    """Setup logging with the given configuration."""
    from ..logging.logger import setup_logging as _setup_logging
    return _setup_logging(**kwargs)

__all__ = ['Config', 'get_config', 'Settings', 'get_settings', 'setup_logging']

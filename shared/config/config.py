"""Configuration management for the stock analyzer application."""
import os
from typing import Optional
from .settings import Settings


class Config:
    """Configuration manager for the application."""
    
    _instance: Optional['Config'] = None
    _settings: Optional[Settings] = None
    
    def __new__(cls) -> 'Config':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager."""
        if self._settings is None:
            self._load_settings()
    
    def _load_settings(self) -> None:
        """Load settings from environment variables and defaults."""
        self._settings = Settings.from_env()
        self._settings.validate()
    
    @property
    def settings(self) -> Settings:
        """Get the current settings."""
        if self._settings is None:
            self._load_settings()
        assert self._settings is not None
        return self._settings
    
    def reload(self) -> None:
        """Reload configuration from environment."""
        self._settings = None
        self._load_settings()
    
    def update_settings(self, **kwargs) -> None:
        """Update specific settings programmatically."""
        if self._settings is None:
            self._load_settings()
        
        # Update settings based on provided kwargs
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
            else:
                # Handle nested settings
                parts = key.split('.')
                if len(parts) == 2:
                    section, setting = parts
                    if hasattr(self._settings, section):
                        section_obj = getattr(self._settings, section)
                        if hasattr(section_obj, setting):
                            setattr(section_obj, setting, value)
        
        # Re-validate after updates
        if self._settings is not None:
            self._settings.validate()


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def get_settings() -> Settings:
    """Get the current settings."""
    return get_config().settings


# Convenience functions for common settings
def get_db_path() -> str:
    """Get the database path."""
    return get_settings().database.path


def get_plots_path() -> str:
    """Get the plots directory path."""
    return get_settings().plotting.plots_path


def get_timezone() -> str:
    """Get the configured timezone."""
    return get_settings().timezone


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return get_settings().debug


def get_log_level() -> int:
    """Get the configured log level."""
    return get_settings().logging.log_level

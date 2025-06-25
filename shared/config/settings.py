"""Application settings and configuration values."""
import os
from dataclasses import dataclass
from typing import Optional
import pytz


@dataclass
class DatabaseSettings:
    """Database configuration settings."""
    name: str = 'data/stocks.db'
    connection_pool_size: int = 10
    timeout: int = 30
    
    @property
    def path(self) -> str:
        """Get the full database path."""
        return os.path.abspath(self.name)


@dataclass
class ScrapingSettings:
    """Data scraping configuration settings."""
    max_workers: int = 10
    batch_size: int = 5
    retry_attempts: int = 5
    retry_delay: float = 0.05
    rate_limit_delay: float = 0.1
    timeout: int = 30
    user_agent: str = "yFinance-Stock-Analyzer/1.0"


@dataclass
class AnalysisSettings:
    """Stock analysis configuration settings."""
    default_lookback_months: int = 6
    min_data_points: int = 50
    rsi_period: int = 14
    volatility_period: int = 20
    ma_short_period: int = 20
    ma_long_period: int = 50
    momentum_period: int = 7
    volume_period: int = 20
    
    # Signal thresholds
    rsi_oversold: float = 40.0
    rsi_overbought: float = 70.0
    volatility_low_threshold: float = 0.3
    volatility_high_threshold: float = 0.5
    momentum_threshold: float = 0.02
    volume_surge_threshold: float = 1.3
    
    # Scoring thresholds
    buy_score_threshold: int = 5
    sell_score_threshold: int = 5
    min_buy_conditions: int = 4
    min_sell_conditions: int = 3


@dataclass
class NewsSettings:
    """News analysis configuration settings."""
    default_days_back: int = 7
    sentiment_days_back: int = 7
    max_news_items: int = 100
    sentiment_positive_threshold: float = 0.1
    sentiment_negative_threshold: float = -0.1
    
    # Sentiment analysis models
    use_finbert: bool = False
    use_roberta: bool = False
    fallback_to_textblob: bool = True


@dataclass
class PlottingSettings:
    """Plotting and visualization settings."""
    plots_path: str = 'plots'
    figure_width: int = 1920
    figure_height: int = 1080
    dpi: int = 100
    save_format: str = 'html'
    
    # Chart settings
    chart_height: int = 900
    subplot_spacing: float = 0.04
    
    @property
    def stock_analysis_path(self) -> str:
        """Get the stock analysis plots directory."""
        return os.path.join(self.plots_path, 'stock_analysis')


@dataclass
class LoggingSettings:
    """Logging configuration settings."""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    @property
    def log_level(self) -> int:
        """Get the numeric log level."""
        import logging
        return getattr(logging, self.level.upper(), logging.INFO)


@dataclass
class Settings:
    """Main application settings container."""
    database: DatabaseSettings
    scraping: ScrapingSettings
    analysis: AnalysisSettings
    news: NewsSettings
    plotting: PlottingSettings
    logging: LoggingSettings
    
    # General settings
    timezone: str = "US/Mountain"
    environment: str = "development"
    debug: bool = False
    
    @property
    def tz(self) -> pytz.BaseTzInfo:
        """Get the timezone object."""
        return pytz.timezone(self.timezone)
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        return cls(
            database=DatabaseSettings(
                name=os.getenv('DB_NAME', 'data/stocks.db'),
                connection_pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
                timeout=int(os.getenv('DB_TIMEOUT', '30'))
            ),
            scraping=ScrapingSettings(
                max_workers=int(os.getenv('SCRAPING_MAX_WORKERS', '10')),
                batch_size=int(os.getenv('SCRAPING_BATCH_SIZE', '5')),
                retry_attempts=int(os.getenv('SCRAPING_RETRY_ATTEMPTS', '5')),
                timeout=int(os.getenv('SCRAPING_TIMEOUT', '30'))
            ),
            analysis=AnalysisSettings(
                default_lookback_months=int(os.getenv('ANALYSIS_LOOKBACK_MONTHS', '6')),
                min_data_points=int(os.getenv('ANALYSIS_MIN_DATA_POINTS', '50')),
                buy_score_threshold=int(os.getenv('ANALYSIS_BUY_THRESHOLD', '5')),
                sell_score_threshold=int(os.getenv('ANALYSIS_SELL_THRESHOLD', '5'))
            ),
            news=NewsSettings(
                default_days_back=int(os.getenv('NEWS_DAYS_BACK', '7')),
                max_news_items=int(os.getenv('NEWS_MAX_ITEMS', '100')),
                use_finbert=os.getenv('NEWS_USE_FINBERT', 'false').lower() == 'true',
                use_roberta=os.getenv('NEWS_USE_ROBERTA', 'false').lower() == 'true'
            ),
            plotting=PlottingSettings(
                plots_path=os.getenv('PLOTS_PATH', 'plots'),
                figure_width=int(os.getenv('PLOT_WIDTH', '1920')),
                figure_height=int(os.getenv('PLOT_HEIGHT', '1080'))
            ),
            logging=LoggingSettings(
                level=os.getenv('LOG_LEVEL', 'INFO'),
                file_path=os.getenv('LOG_FILE_PATH'),
                max_file_size=int(os.getenv('LOG_MAX_SIZE', str(10 * 1024 * 1024)))
            ),
            timezone=os.getenv('TIMEZONE', 'US/Mountain'),
            environment=os.getenv('ENVIRONMENT', 'development'),
            debug=os.getenv('DEBUG', 'false').lower() == 'true'
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate database settings
        if not self.database.name:
            errors.append("Database name cannot be empty")
        
        # Validate scraping settings
        if self.scraping.max_workers <= 0:
            errors.append("Scraping max_workers must be positive")
        
        if self.scraping.batch_size <= 0:
            errors.append("Scraping batch_size must be positive")
        
        # Validate analysis settings
        if self.analysis.min_data_points < 20:
            errors.append("Analysis min_data_points must be at least 20")
        
        if self.analysis.rsi_period <= 0:
            errors.append("RSI period must be positive")
        
        # Validate timezone
        try:
            pytz.timezone(self.timezone)
        except pytz.UnknownTimeZoneError:
            errors.append(f"Unknown timezone: {self.timezone}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

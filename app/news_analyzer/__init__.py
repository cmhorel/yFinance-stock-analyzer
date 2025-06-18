# news_analyzer/__init__.py
"""
News analysis module for stock sentiment and industry analysis.
"""
import logging
from .news_processor import NewsProcessor
from .industry_analyzer import IndustryAnalyzer
from app.database_manager import db_manager  # NEW: Import centralized database manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create global instances
_news_processor = None
_industry_analyzer = None

def get_news_processor() -> NewsProcessor:
    """Get singleton NewsProcessor instance."""
    global _news_processor
    if _news_processor is None:
        _news_processor = NewsProcessor()
    return _news_processor

def get_industry_analyzer() -> IndustryAnalyzer:
    """Get singleton IndustryAnalyzer instance."""
    global _industry_analyzer
    if _industry_analyzer is None:
        _industry_analyzer = IndustryAnalyzer()
    return _industry_analyzer

# Public API functions for backward compatibility
def get_industry_and_sector(ticker: str):
    """Fetch industry and sector for a ticker."""
    from .data_fetcher import DataFetcher
    return DataFetcher.get_stock_info(ticker)

def fetch_recent_news(ticker: str, days_back: int = 7):
    """Fetch recent news for a ticker."""
    from .data_fetcher import DataFetcher
    return DataFetcher.fetch_recent_news(ticker, days_back)

def analyze_news_sentiment(news_items):
    """Analyze sentiment for news items."""
    processor = get_news_processor()
    return processor._analyze_news_sentiments(news_items)

def store_industry_and_news(ticker: str, stock_id: int, industry_data, news_sentiments, c=None):
    """Store industry and news data using database manager."""
    if industry_data:
        db_manager.store_stock_info(stock_id, industry_data['sector'], industry_data['industry'])
    if news_sentiments:
        db_manager.store_news_sentiments(stock_id, news_sentiments)

def get_average_sentiment(stock_id: int, days_back: int = 7) -> float:
    """Get average sentiment for a stock using database manager."""
    return db_manager.get_average_sentiment(stock_id, days_back)

def get_sentiment_timeseries(stock_id: int, days_back: int = 60):
    """Get sentiment time series for a stock using database manager."""
    return db_manager.get_sentiment_timeseries(stock_id, days_back)

def get_industry_average_momentum(industry: str, exclude_stock_id: int, df_all):
    """Get industry average momentum."""
    analyzer = get_industry_analyzer()
    return analyzer.get_industry_average_momentum(industry, exclude_stock_id, df_all)

# New simplified API
def process_stock_news(ticker: str, stock_id: int, days_back: int = 7, analyze_sentiment: bool = True):
    """Process all news and sentiment for a stock."""
    processor = get_news_processor()
    processor.process_news_for_stock(ticker, stock_id, days_back, analyze_sentiment=analyze_sentiment)
# news_analyzer/data_fetcher.py
"""
Data fetching module for stock information and news.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import yfinance as yf
import curl_cffi.requests as requests
from app import appconfig  # Importing TIME_ZONE from appconfig
import pytz

TIME_ZONE = pytz.timezone(appconfig.TIME_ZONE)
YAHOO_TIME_ZONE = pytz.timezone("America/New_York")  # Default timezone for Yahoo Finance

yt_session = requests.Session(impersonate="chrome")

logger = logging.getLogger(__name__)

class DataFetcher:
    """Handles fetching of stock data and news from external sources."""
    
    @staticmethod
    def get_stock_info(ticker: str) -> Optional[Dict[str, str]]:
        """
        Fetch industry and sector information for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with 'sector' and 'industry' keys, or None if failed
        """
        try:
            info = yf.Ticker(ticker).info
            return {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return None
    
    @staticmethod
    def fetch_recent_news(ticker: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch recent news for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back for news
            
        Returns:
            List of news items
        """
        try:
            cutoff_date = datetime.now(TIME_ZONE) - timedelta(days=days_back)
            news = yf.Ticker(ticker, session=yt_session).news
            
            recent_news = []
            for item in news:
                try:
                    pub_date = datetime.strptime(
                        item['content']['pubDate'], 
                        '%Y-%m-%dT%H:%M:%SZ'
                    ).replace(tzinfo=YAHOO_TIME_ZONE)
                    if pub_date >= cutoff_date:
                        recent_news.append(item)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error parsing news item: {e}")
                    continue
            
            return recent_news
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
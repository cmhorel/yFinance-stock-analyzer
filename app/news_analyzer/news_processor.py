# news_analyzer/news_processor.py
"""
News processing and sentiment analysis coordination.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any
from .sentiment_analyzer import SentimentAnalyzer
from .data_fetcher import DataFetcher
from app.database_manager import DatabaseManager
from app import appconfig
import pytz

TIME_ZONE = pytz.timezone(appconfig.TIME_ZONE)
YAHOO_TIME_ZONE = pytz.timezone("America/New_York")  # Default timezone for Yahoo Finance

logger = logging.getLogger(__name__)

class NewsProcessor:
    """Coordinates news fetching, sentiment analysis, and storage."""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.data_fetcher = DataFetcher()
        self.db = DatabaseManager()

    def get_sentiment_analyzer(self) -> SentimentAnalyzer:
        """Get the sentiment analyzer instance."""
        if not self.sentiment_analyzer:
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("Initialized SentimentAnalyzer")
        return self.sentiment_analyzer

    def process_news_for_stock(self, ticker: str, stock_id: int, days_back: int = 7, analyze_sentiment: bool = True) -> None:
        """
        Process news and sentiment for a single stock.

        Args:
            ticker: Stock ticker symbol
            stock_id: Database stock ID
            days_back: Days to look back for news
            analyze_sentiment: Whether to analyze sentiment now or defer
        """
        try:
            # Fetch stock info
            stock_info = self.data_fetcher.get_stock_info(ticker)
            if stock_info:
                self.db.store_stock_info(stock_id, stock_info['sector'], stock_info['industry'])
                logger.info(f"Stored info for {ticker}: {stock_info['sector']}, {stock_info['industry']}")

            # Fetch news
            news_items = self.data_fetcher.fetch_recent_news(ticker, days_back)
            if news_items:
                if analyze_sentiment:
                    sentiments = self._analyze_news_sentiments(news_items)
                else:
                    # Store news with sentiment_score as None
                    sentiments = []
                    for item in news_items:
                        content = item.get('content', {})
                        title = content.get('title', '') or 'N/A'
                        summary = content.get('summary', '') or 'N/A'
                        pub_date = 'N/A'
                        if 'pubDate' in content:
                            try:
                                pub_date = datetime.strptime(content['pubDate'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=YAHOO_TIME_ZONE)
                            except ValueError:
                                pub_date = datetime.now(TIME_ZONE)
                        sentiments.append({
                            'title': title,
                            'summary': summary,
                            'publish_date': pub_date,
                            'sentiment_score': None
                        })
                self.db.store_news_sentiments(stock_id, sentiments)
                logger.info(f"Processed {len(sentiments)} news items for {ticker}")
            else:
                logger.info(f"No recent news found for {ticker}")

        except Exception as e:
            logger.error(f"Error processing news for {ticker}: {e}")
    
    def _analyze_news_sentiments(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for news items."""
        sentiments = []
        
        for item in news_items:
            try:
                content = item.get('content', {})
                title = content.get('title', '')
                summary = content.get('summary', '')
                provider = content.get('provider', {}).get('displayName', '')
                
                # Combine text for analysis
                text = f"{title} {provider} {summary}".strip()
                
                # Analyze sentiment
                sentiment_score = self.get_sentiment_analyzer().analyze(text)

                # Parse publish date
                pub_date = 'N/A'
                if 'pubDate' in content:
                    try:
                        pub_date = datetime.strptime(content['pubDate'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=YAHOO_TIME_ZONE)
                    except ValueError:
                        pub_date = datetime.now(TIME_ZONE)
                
                sentiments.append({
                    'title': title or 'N/A',
                    'summary': summary or 'N/A',
                    'publish_date': pub_date,
                    'sentiment_score': sentiment_score
                })
                
            except Exception as e:
                logger.error(f"Error analyzing news item: {e}")
                continue
        
        return sentiments
    
    


    def get_average_sentiment(self, stock_id: int, days_back: int = 7) -> float:
        """Get average sentiment for a stock."""
        return self.db.get_average_sentiment(stock_id, days_back)
    
    def get_sentiment_timeseries(self, stock_id: int, days_back: int = 60):
        """Get sentiment time series for a stock."""
        return self.db.get_sentiment_timeseries(stock_id, days_back)


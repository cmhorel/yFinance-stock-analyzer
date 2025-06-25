"""News repository interface."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from decimal import Decimal

from ..entities.news import NewsItem, SentimentScore, SentimentType


class INewsRepository(ABC):
    """Interface for news data repository operations."""
    
    # News Item Operations
    @abstractmethod
    async def get_news_by_id(self, news_id: int) -> Optional[NewsItem]:
        """Get a news item by its ID."""
        pass
    
    @abstractmethod
    async def get_news_for_stock(
        self, 
        stock_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[NewsItem]:
        """Get news items for a specific stock."""
        pass
    
    @abstractmethod
    async def get_recent_news(
        self, 
        days_back: int = 7,
        limit: Optional[int] = None
    ) -> List[NewsItem]:
        """Get recent news items across all stocks."""
        pass
    
    @abstractmethod
    async def create_news_item(self, news_item: NewsItem) -> NewsItem:
        """Create a new news item."""
        pass
    
    @abstractmethod
    async def bulk_create_news_items(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Create multiple news items in bulk."""
        pass
    
    @abstractmethod
    async def update_news_item(self, news_item: NewsItem) -> NewsItem:
        """Update an existing news item."""
        pass
    
    @abstractmethod
    async def delete_news_item(self, news_id: int) -> bool:
        """Delete a news item."""
        pass
    
    @abstractmethod
    async def news_exists(self, stock_id: int, title: str, published_date: date) -> bool:
        """Check if a news item already exists to avoid duplicates."""
        pass
    
    # Sentiment Score Operations
    @abstractmethod
    async def get_sentiment_by_id(self, sentiment_id: int) -> Optional[SentimentScore]:
        """Get a sentiment score by its ID."""
        pass
    
    @abstractmethod
    async def get_sentiment_for_news(self, news_id: int) -> Optional[SentimentScore]:
        """Get sentiment score for a specific news item."""
        pass
    
    @abstractmethod
    async def get_sentiments_for_stock(
        self, 
        stock_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[SentimentScore]:
        """Get sentiment scores for a stock's news items."""
        pass
    
    @abstractmethod
    async def create_sentiment_score(self, sentiment: SentimentScore) -> SentimentScore:
        """Create a new sentiment score."""
        pass
    
    @abstractmethod
    async def bulk_create_sentiment_scores(self, sentiments: List[SentimentScore]) -> List[SentimentScore]:
        """Create multiple sentiment scores in bulk."""
        pass
    
    @abstractmethod
    async def update_sentiment_score(self, sentiment: SentimentScore) -> SentimentScore:
        """Update an existing sentiment score."""
        pass
    
    @abstractmethod
    async def delete_sentiment_score(self, sentiment_id: int) -> bool:
        """Delete a sentiment score."""
        pass
    
    # Analytics and Aggregations
    @abstractmethod
    async def get_average_sentiment(
        self, 
        stock_id: int, 
        days_back: int = 7
    ) -> Optional[Decimal]:
        """Get average sentiment score for a stock over specified days."""
        pass
    
    @abstractmethod
    async def get_sentiment_distribution(
        self, 
        stock_id: int, 
        days_back: int = 30
    ) -> Dict[SentimentType, int]:
        """Get distribution of sentiment types for a stock."""
        pass
    
    @abstractmethod
    async def get_sentiment_trend(
        self, 
        stock_id: int, 
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get daily sentiment trend for a stock."""
        pass
    
    @abstractmethod
    async def get_news_volume_by_stock(
        self, 
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """Get news volume statistics by stock."""
        pass
    
    @abstractmethod
    async def get_most_mentioned_stocks(
        self, 
        days_back: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get stocks with the most news mentions."""
        pass
    
    @abstractmethod
    async def get_sentiment_leaders(
        self, 
        days_back: int = 7,
        sentiment_type: SentimentType = SentimentType.POSITIVE,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get stocks with highest sentiment scores."""
        pass
    
    # News without sentiment (for processing)
    @abstractmethod
    async def get_news_without_sentiment(
        self, 
        limit: Optional[int] = None
    ) -> List[NewsItem]:
        """Get news items that don't have sentiment scores yet."""
        pass
    
    @abstractmethod
    async def get_outdated_sentiments(
        self, 
        days_old: int = 30,
        limit: Optional[int] = None
    ) -> List[SentimentScore]:
        """Get sentiment scores that might need re-analysis."""
        pass
    
    # Search and filtering
    @abstractmethod
    async def search_news(
        self, 
        query: str,
        stock_id: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[NewsItem]:
        """Search news items by text query."""
        pass
    
    @abstractmethod
    async def get_news_by_source(
        self, 
        source: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[NewsItem]:
        """Get news items from a specific source."""
        pass
    
    @abstractmethod
    async def get_news_sources(self) -> List[str]:
        """Get all unique news sources."""
        pass
    
    # Cleanup operations
    @abstractmethod
    async def cleanup_old_news(self, days_old: int = 365) -> int:
        """Remove news items older than specified days. Returns count of deleted items."""
        pass
    
    @abstractmethod
    async def cleanup_duplicate_news(self) -> int:
        """Remove duplicate news items. Returns count of deleted items."""
        pass

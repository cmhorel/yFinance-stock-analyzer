"""News-related domain entities."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from decimal import Decimal
from enum import Enum


class SentimentType(Enum):
    """Enumeration for sentiment types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class NewsItem:
    """Represents a news item related to a stock."""
    id: Optional[int]
    stock_id: int
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    published_date: Optional[datetime] = None
    scraped_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate news item data after initialization."""
        if not self.title or not self.title.strip():
            raise ValueError("News title cannot be empty")
        self.title = self.title.strip()
        
        if self.summary:
            self.summary = self.summary.strip()
        if self.content:
            self.content = self.content.strip()
        if self.source:
            self.source = self.source.strip()
        if self.author:
            self.author = self.author.strip()
    
    @property
    def is_recent(self, days: int = 7) -> bool:
        """Check if the news item is recent (within specified days)."""
        if not self.published_date:
            return False
        return (datetime.now() - self.published_date).days <= days


@dataclass
class SentimentScore:
    """Represents sentiment analysis results for a news item."""
    id: Optional[int]
    news_id: int
    score: Decimal  # Range: -1.0 to 1.0
    confidence: Optional[Decimal] = None  # Range: 0.0 to 1.0
    sentiment_type: Optional[SentimentType] = None
    analyzer_model: Optional[str] = None
    analyzed_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate sentiment score data after initialization."""
        if not (-1 <= self.score <= 1):
            raise ValueError("Sentiment score must be between -1.0 and 1.0")
        
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        # Auto-determine sentiment type if not provided
        if self.sentiment_type is None:
            if self.score > 0.1:
                self.sentiment_type = SentimentType.POSITIVE
            elif self.score < -0.1:
                self.sentiment_type = SentimentType.NEGATIVE
            else:
                self.sentiment_type = SentimentType.NEUTRAL
    
    @property
    def is_positive(self) -> bool:
        """Check if sentiment is positive."""
        return self.sentiment_type == SentimentType.POSITIVE
    
    @property
    def is_negative(self) -> bool:
        """Check if sentiment is negative."""
        return self.sentiment_type == SentimentType.NEGATIVE
    
    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return self.sentiment_type == SentimentType.NEUTRAL
    
    @property
    def strength(self) -> str:
        """Get sentiment strength description."""
        abs_score = abs(self.score)
        if abs_score >= 0.7:
            return "strong"
        elif abs_score >= 0.3:
            return "moderate"
        else:
            return "weak"

"""Stock-related domain entities."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from decimal import Decimal


@dataclass
class Stock:
    """Represents a stock entity."""
    id: Optional[int]
    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate stock data after initialization."""
        if not self.symbol:
            raise ValueError("Stock symbol cannot be empty")
        self.symbol = self.symbol.upper().strip()


@dataclass
class StockPrice:
    """Represents stock price data for a specific date."""
    id: Optional[int]
    stock_id: int
    date: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int
    adjusted_close: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate price data after initialization."""
        if self.high_price < self.low_price:
            raise ValueError("High price cannot be less than low price")
        if self.open_price < 0 or self.close_price < 0:
            raise ValueError("Prices cannot be negative")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def price_range(self) -> Decimal:
        """Calculate the price range for the day."""
        return self.high_price - self.low_price
    
    @property
    def daily_return(self) -> Decimal:
        """Calculate daily return percentage."""
        if self.open_price == 0:
            return Decimal('0')
        return ((self.close_price - self.open_price) / self.open_price) * 100


@dataclass
class StockInfo:
    """Represents additional stock information."""
    id: Optional[int]
    stock_id: int
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[Decimal] = None
    pe_ratio: Optional[Decimal] = None
    dividend_yield: Optional[Decimal] = None
    beta: Optional[Decimal] = None
    description: Optional[str] = None
    website: Optional[str] = None
    employees: Optional[int] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Clean and validate stock info data."""
        if self.sector:
            self.sector = self.sector.strip()
        if self.industry:
            self.industry = self.industry.strip()
        if self.description:
            self.description = self.description.strip()

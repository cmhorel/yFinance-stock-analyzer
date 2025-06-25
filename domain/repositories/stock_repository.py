"""Stock repository interface."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from decimal import Decimal

from ..entities.stock import Stock, StockPrice, StockInfo


class IStockRepository(ABC):
    """Interface for stock data repository operations."""
    
    @abstractmethod
    async def get_stock_by_id(self, stock_id: int) -> Optional[Stock]:
        """Get a stock by its ID."""
        pass
    
    @abstractmethod
    async def get_stock_by_symbol(self, symbol: str) -> Optional[Stock]:
        """Get a stock by its symbol."""
        pass
    
    @abstractmethod
    async def get_all_stocks(self) -> List[Stock]:
        """Get all stocks."""
        pass
    
    @abstractmethod
    async def create_stock(self, stock: Stock) -> Stock:
        """Create a new stock record."""
        pass
    
    @abstractmethod
    async def update_stock(self, stock: Stock) -> Stock:
        """Update an existing stock record."""
        pass
    
    @abstractmethod
    async def delete_stock(self, stock_id: int) -> bool:
        """Delete a stock record."""
        pass
    
    @abstractmethod
    async def get_or_create_stock(self, symbol: str) -> Stock:
        """Get existing stock or create new one if not found."""
        pass
    
    # Stock Price Operations
    @abstractmethod
    async def get_stock_prices(
        self, 
        stock_id: int, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[StockPrice]:
        """Get stock prices for a given stock within date range."""
        pass
    
    @abstractmethod
    async def get_latest_stock_price(self, stock_id: int) -> Optional[StockPrice]:
        """Get the most recent stock price."""
        pass
    
    @abstractmethod
    async def get_latest_stock_price_date(self, stock_id: int) -> Optional[date]:
        """Get the date of the most recent stock price."""
        pass
    
    @abstractmethod
    async def create_stock_prices(self, prices: List[StockPrice]) -> List[StockPrice]:
        """Create multiple stock price records."""
        pass
    
    @abstractmethod
    async def get_stock_price_on_date(self, stock_id: int, target_date: date) -> Optional[StockPrice]:
        """Get stock price for a specific date."""
        pass
    
    @abstractmethod
    async def get_stocks_with_prices_since(self, since_date: date) -> List[Dict[str, Any]]:
        """Get stocks that have price data since the given date."""
        pass
    
    # Stock Info Operations
    @abstractmethod
    async def get_stock_info(self, stock_id: int) -> Optional[StockInfo]:
        """Get stock information (sector, industry, etc.)."""
        pass
    
    @abstractmethod
    async def create_or_update_stock_info(self, stock_info: StockInfo) -> StockInfo:
        """Create or update stock information."""
        pass
    
    @abstractmethod
    async def get_stocks_by_sector(self, sector: str) -> List[Stock]:
        """Get all stocks in a specific sector."""
        pass
    
    @abstractmethod
    async def get_stocks_by_industry(self, industry: str) -> List[Stock]:
        """Get all stocks in a specific industry."""
        pass
    
    @abstractmethod
    async def get_sectors(self) -> List[str]:
        """Get all unique sectors."""
        pass
    
    @abstractmethod
    async def get_industries(self) -> List[str]:
        """Get all unique industries."""
        pass
    
    # Bulk Operations
    @abstractmethod
    async def bulk_create_stocks(self, stocks: List[Stock]) -> List[Stock]:
        """Create multiple stock records in bulk."""
        pass
    
    @abstractmethod
    async def bulk_update_stock_prices(self, prices: List[StockPrice]) -> int:
        """Update multiple stock prices in bulk. Returns count of updated records."""
        pass
    
    # Analytics and Aggregations
    @abstractmethod
    async def get_stock_statistics(
        self, 
        stock_id: int, 
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get statistical data for a stock (avg price, volatility, etc.)."""
        pass
    
    @abstractmethod
    async def get_top_performers(
        self, 
        days_back: int = 30, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top performing stocks by percentage gain."""
        pass
    
    @abstractmethod
    async def get_most_volatile_stocks(
        self, 
        days_back: int = 30, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get most volatile stocks by price variance."""
        pass
    
    @abstractmethod
    async def get_stocks_for_analysis(
        self, 
        months_back: int = 6,
        min_data_points: int = 50
    ) -> List[Dict[str, Any]]:
        """Get stocks with sufficient data for analysis."""
        pass

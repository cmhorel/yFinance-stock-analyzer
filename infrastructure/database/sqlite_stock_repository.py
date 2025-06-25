"""SQLite implementation of the stock repository."""
import sqlite3
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from decimal import Decimal
from contextlib import asynccontextmanager

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from domain.repositories.stock_repository import IStockRepository
from domain.entities.stock import Stock, StockPrice, StockInfo
from shared.config import get_settings
from shared.logging import get_logger
from shared.exceptions.data import DataNotFoundError, DataValidationError


class SqliteStockRepository(IStockRepository):
    """SQLite implementation of stock repository."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.settings = get_settings()
        self.db_path = db_path or self.settings.database.path
        self.logger = get_logger(__name__)
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    name TEXT,
                    exchange TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS stock_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    open_price DECIMAL(10,4) NOT NULL,
                    high_price DECIMAL(10,4) NOT NULL,
                    low_price DECIMAL(10,4) NOT NULL,
                    close_price DECIMAL(10,4) NOT NULL,
                    volume INTEGER NOT NULL,
                    adjusted_close DECIMAL(10,4),
                    FOREIGN KEY (stock_id) REFERENCES stocks (id),
                    UNIQUE(stock_id, date)
                );
                
                CREATE TABLE IF NOT EXISTS stock_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_id INTEGER NOT NULL,
                    sector TEXT,
                    industry TEXT,
                    market_cap DECIMAL(15,2),
                    pe_ratio DECIMAL(8,2),
                    dividend_yield DECIMAL(6,4),
                    beta DECIMAL(6,4),
                    description TEXT,
                    website TEXT,
                    employees INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (stock_id) REFERENCES stocks (id),
                    UNIQUE(stock_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(date);
                CREATE INDEX IF NOT EXISTS idx_stock_prices_stock_date ON stock_prices(stock_id, date);
                CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
            """)
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _row_to_stock(self, row: sqlite3.Row) -> Stock:
        """Convert database row to Stock entity."""
        return Stock(
            id=row['id'],
            symbol=row['symbol'],
            name=row['name'],
            exchange=row['exchange'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        )
    
    def _row_to_stock_price(self, row: sqlite3.Row) -> StockPrice:
        """Convert database row to StockPrice entity."""
        return StockPrice(
            id=row['id'],
            stock_id=row['stock_id'],
            date=datetime.fromisoformat(row['date']),
            open_price=Decimal(str(row['open_price'])),
            high_price=Decimal(str(row['high_price'])),
            low_price=Decimal(str(row['low_price'])),
            close_price=Decimal(str(row['close_price'])),
            volume=row['volume'],
            adjusted_close=Decimal(str(row['adjusted_close'])) if row['adjusted_close'] else None
        )
    
    def _row_to_stock_info(self, row: sqlite3.Row) -> StockInfo:
        """Convert database row to StockInfo entity."""
        return StockInfo(
            id=row['id'],
            stock_id=row['stock_id'],
            sector=row['sector'],
            industry=row['industry'],
            market_cap=Decimal(str(row['market_cap'])) if row['market_cap'] else None,
            pe_ratio=Decimal(str(row['pe_ratio'])) if row['pe_ratio'] else None,
            dividend_yield=Decimal(str(row['dividend_yield'])) if row['dividend_yield'] else None,
            beta=Decimal(str(row['beta'])) if row['beta'] else None,
            description=row['description'],
            website=row['website'],
            employees=row['employees'],
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        )
    
    # Stock Operations
    async def get_stock_by_id(self, stock_id: int) -> Optional[Stock]:
        """Get a stock by its ID."""
        async with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM stocks WHERE id = ?", (stock_id,))
            row = cursor.fetchone()
            return self._row_to_stock(row) if row else None
    
    async def get_stock_by_symbol(self, symbol: str) -> Optional[Stock]:
        """Get a stock by its symbol."""
        async with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM stocks WHERE symbol = ?", (symbol.upper(),))
            row = cursor.fetchone()
            return self._row_to_stock(row) if row else None
    
    async def get_all_stocks(self) -> List[Stock]:
        """Get all stocks."""
        async with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM stocks ORDER BY symbol")
            return [self._row_to_stock(row) for row in cursor.fetchall()]
    
    async def create_stock(self, stock: Stock) -> Stock:
        """Create a new stock record."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO stocks (symbol, name, exchange, created_at, updated_at) 
                   VALUES (?, ?, ?, ?, ?)""",
                (stock.symbol.upper(), stock.name, stock.exchange, 
                 datetime.now().isoformat(), datetime.now().isoformat())
            )
            conn.commit()
            
            stock.id = cursor.lastrowid
            stock.created_at = datetime.now()
            stock.updated_at = datetime.now()
            
            self.logger.info(f"Created stock: {stock.symbol} (ID: {stock.id})")
            return stock
    
    async def update_stock(self, stock: Stock) -> Stock:
        """Update an existing stock record."""
        async with self._get_connection() as conn:
            conn.execute(
                """UPDATE stocks SET name = ?, exchange = ?, updated_at = ? 
                   WHERE id = ?""",
                (stock.name, stock.exchange, datetime.now().isoformat(), stock.id)
            )
            conn.commit()
            
            stock.updated_at = datetime.now()
            self.logger.info(f"Updated stock: {stock.symbol} (ID: {stock.id})")
            return stock
    
    async def delete_stock(self, stock_id: int) -> bool:
        """Delete a stock record."""
        async with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM stocks WHERE id = ?", (stock_id,))
            conn.commit()
            
            deleted = cursor.rowcount > 0
            if deleted:
                self.logger.info(f"Deleted stock ID: {stock_id}")
            return deleted
    
    async def get_or_create_stock(self, symbol: str) -> Stock:
        """Get existing stock or create new one if not found."""
        stock = await self.get_stock_by_symbol(symbol)
        if stock:
            return stock
        
        # Create new stock with minimal info
        new_stock = Stock(
            id=None,
            symbol=symbol.upper(),
            name=None,
            exchange=None
        )
        return await self.create_stock(new_stock)
    
    # Stock Price Operations
    async def get_stock_prices(
        self, 
        stock_id: int, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[StockPrice]:
        """Get stock prices for a given stock within date range."""
        query = "SELECT * FROM stock_prices WHERE stock_id = ?"
        params: List[Any] = [stock_id]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY date DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        async with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_stock_price(row) for row in cursor.fetchall()]
    
    async def get_latest_stock_price(self, stock_id: int) -> Optional[StockPrice]:
        """Get the most recent stock price."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM stock_prices WHERE stock_id = ? ORDER BY date DESC LIMIT 1",
                (stock_id,)
            )
            row = cursor.fetchone()
            return self._row_to_stock_price(row) if row else None
    
    async def get_latest_stock_price_date(self, stock_id: int) -> Optional[date]:
        """Get the date of the most recent stock price."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT date FROM stock_prices WHERE stock_id = ? ORDER BY date DESC LIMIT 1",
                (stock_id,)
            )
            row = cursor.fetchone()
            return datetime.fromisoformat(row['date']).date() if row else None
    
    async def create_stock_prices(self, prices: List[StockPrice]) -> List[StockPrice]:
        """Create multiple stock price records."""
        if not prices:
            return []
        
        async with self._get_connection() as conn:
            for price in prices:
                try:
                    cursor = conn.execute(
                        """INSERT OR REPLACE INTO stock_prices 
                           (stock_id, date, open_price, high_price, low_price, 
                            close_price, volume, adjusted_close) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (price.stock_id, price.date.isoformat(), 
                         float(price.open_price), float(price.high_price),
                         float(price.low_price), float(price.close_price),
                         price.volume, 
                         float(price.adjusted_close) if price.adjusted_close else None)
                    )
                    if price.id is None:
                        price.id = cursor.lastrowid
                except Exception as e:
                    self.logger.error(f"Error inserting price for stock {price.stock_id}: {e}")
                    raise DataValidationError(f"Failed to insert price data: {e}")
            
            conn.commit()
            self.logger.info(f"Created {len(prices)} stock price records")
            return prices
    
    async def get_stock_price_on_date(self, stock_id: int, target_date: date) -> Optional[StockPrice]:
        """Get stock price for a specific date."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM stock_prices WHERE stock_id = ? AND date = ?",
                (stock_id, target_date.isoformat())
            )
            row = cursor.fetchone()
            return self._row_to_stock_price(row) if row else None
    
    async def get_stocks_with_prices_since(self, since_date: date) -> List[Dict[str, Any]]:
        """Get stocks that have price data since the given date."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT s.*, COUNT(sp.id) as price_count, MAX(sp.date) as latest_date
                   FROM stocks s 
                   LEFT JOIN stock_prices sp ON s.id = sp.stock_id 
                   WHERE sp.date >= ? 
                   GROUP BY s.id 
                   ORDER BY latest_date DESC""",
                (since_date.isoformat(),)
            )
            
            results = []
            for row in cursor.fetchall():
                stock = self._row_to_stock(row)
                results.append({
                    'stock': stock,
                    'price_count': row['price_count'],
                    'latest_date': datetime.fromisoformat(row['latest_date']).date() if row['latest_date'] else None
                })
            
            return results
    
    # Stock Info Operations
    async def get_stock_info(self, stock_id: int) -> Optional[StockInfo]:
        """Get stock information (sector, industry, etc.)."""
        async with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM stock_info WHERE stock_id = ?", (stock_id,))
            row = cursor.fetchone()
            return self._row_to_stock_info(row) if row else None
    
    async def create_or_update_stock_info(self, stock_info: StockInfo) -> StockInfo:
        """Create or update stock information."""
        async with self._get_connection() as conn:
            # Try to update first
            cursor = conn.execute(
                """UPDATE stock_info SET 
                   sector = ?, industry = ?, market_cap = ?, pe_ratio = ?,
                   dividend_yield = ?, beta = ?, description = ?, website = ?,
                   employees = ?, updated_at = ?
                   WHERE stock_id = ?""",
                (stock_info.sector, stock_info.industry, 
                 float(stock_info.market_cap) if stock_info.market_cap else None,
                 float(stock_info.pe_ratio) if stock_info.pe_ratio else None,
                 float(stock_info.dividend_yield) if stock_info.dividend_yield else None,
                 float(stock_info.beta) if stock_info.beta else None,
                 stock_info.description, stock_info.website, stock_info.employees,
                 datetime.now().isoformat(), stock_info.stock_id)
            )
            
            if cursor.rowcount == 0:
                # Insert new record
                cursor = conn.execute(
                    """INSERT INTO stock_info 
                       (stock_id, sector, industry, market_cap, pe_ratio,
                        dividend_yield, beta, description, website, employees, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (stock_info.stock_id, stock_info.sector, stock_info.industry,
                     float(stock_info.market_cap) if stock_info.market_cap else None,
                     float(stock_info.pe_ratio) if stock_info.pe_ratio else None,
                     float(stock_info.dividend_yield) if stock_info.dividend_yield else None,
                     float(stock_info.beta) if stock_info.beta else None,
                     stock_info.description, stock_info.website, stock_info.employees,
                     datetime.now().isoformat())
                )
                stock_info.id = cursor.lastrowid
            
            conn.commit()
            stock_info.updated_at = datetime.now()
            
            self.logger.info(f"Updated stock info for stock ID: {stock_info.stock_id}")
            return stock_info
    
    async def get_stocks_by_sector(self, sector: str) -> List[Stock]:
        """Get all stocks in a specific sector."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT s.* FROM stocks s 
                   JOIN stock_info si ON s.id = si.stock_id 
                   WHERE si.sector = ? ORDER BY s.symbol""",
                (sector,)
            )
            return [self._row_to_stock(row) for row in cursor.fetchall()]
    
    async def get_stocks_by_industry(self, industry: str) -> List[Stock]:
        """Get all stocks in a specific industry."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT s.* FROM stocks s 
                   JOIN stock_info si ON s.id = si.stock_id 
                   WHERE si.industry = ? ORDER BY s.symbol""",
                (industry,)
            )
            return [self._row_to_stock(row) for row in cursor.fetchall()]
    
    async def get_sectors(self) -> List[str]:
        """Get all unique sectors."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT sector FROM stock_info WHERE sector IS NOT NULL ORDER BY sector"
            )
            return [row['sector'] for row in cursor.fetchall()]
    
    async def get_industries(self) -> List[str]:
        """Get all unique industries."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT industry FROM stock_info WHERE industry IS NOT NULL ORDER BY industry"
            )
            return [row['industry'] for row in cursor.fetchall()]
    
    # Bulk Operations
    async def bulk_create_stocks(self, stocks: List[Stock]) -> List[Stock]:
        """Create multiple stock records in bulk."""
        created_stocks = []
        for stock in stocks:
            try:
                created_stock = await self.create_stock(stock)
                created_stocks.append(created_stock)
            except Exception as e:
                self.logger.error(f"Error creating stock {stock.symbol}: {e}")
        
        return created_stocks
    
    async def bulk_update_stock_prices(self, prices: List[StockPrice]) -> int:
        """Update multiple stock prices in bulk. Returns count of updated records."""
        return len(await self.create_stock_prices(prices))
    
    # Analytics and Aggregations
    async def get_stock_statistics(
        self, 
        stock_id: int, 
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get statistical data for a stock (avg price, volatility, etc.)."""
        since_date = (datetime.now().date() - timedelta(days=days_back))
        
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT 
                   COUNT(*) as data_points,
                   AVG(close_price) as avg_price,
                   MIN(close_price) as min_price,
                   MAX(close_price) as max_price,
                   AVG(volume) as avg_volume,
                   MIN(date) as start_date,
                   MAX(date) as end_date
                   FROM stock_prices 
                   WHERE stock_id = ? AND date >= ?""",
                (stock_id, since_date.isoformat())
            )
            row = cursor.fetchone()
            
            if row and row['data_points'] > 0:
                return {
                    'stock_id': stock_id,
                    'data_points': row['data_points'],
                    'avg_price': Decimal(str(row['avg_price'])),
                    'min_price': Decimal(str(row['min_price'])),
                    'max_price': Decimal(str(row['max_price'])),
                    'avg_volume': int(row['avg_volume']),
                    'start_date': datetime.fromisoformat(row['start_date']).date(),
                    'end_date': datetime.fromisoformat(row['end_date']).date(),
                    'days_back': days_back
                }
            
            return {'stock_id': stock_id, 'data_points': 0}
    
    async def get_top_performers(
        self, 
        days_back: int = 30, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top performing stocks by percentage gain."""
        since_date = (datetime.now().date() - timedelta(days=days_back))
        
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT s.symbol, s.name,
                   first_price.close_price as start_price,
                   last_price.close_price as end_price,
                   ((last_price.close_price - first_price.close_price) / first_price.close_price * 100) as return_pct
                   FROM stocks s
                   JOIN (
                       SELECT stock_id, close_price, 
                       ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY date ASC) as rn
                       FROM stock_prices WHERE date >= ?
                   ) first_price ON s.id = first_price.stock_id AND first_price.rn = 1
                   JOIN (
                       SELECT stock_id, close_price,
                       ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY date DESC) as rn
                       FROM stock_prices WHERE date >= ?
                   ) last_price ON s.id = last_price.stock_id AND last_price.rn = 1
                   ORDER BY return_pct DESC
                   LIMIT ?""",
                (since_date.isoformat(), since_date.isoformat(), limit)
            )
            
            return [
                {
                    'symbol': row['symbol'],
                    'name': row['name'],
                    'start_price': Decimal(str(row['start_price'])),
                    'end_price': Decimal(str(row['end_price'])),
                    'return_percent': Decimal(str(row['return_pct']))
                }
                for row in cursor.fetchall()
            ]
    
    async def get_most_volatile_stocks(
        self, 
        days_back: int = 30, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get most volatile stocks by price variance."""
        since_date = (datetime.now().date() - timedelta(days=days_back))
        
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT s.symbol, s.name,
                   AVG(sp.close_price) as avg_price,
                   (MAX(sp.close_price) - MIN(sp.close_price)) / AVG(sp.close_price) * 100 as volatility_pct
                   FROM stocks s
                   JOIN stock_prices sp ON s.id = sp.stock_id
                   WHERE sp.date >= ?
                   GROUP BY s.id, s.symbol, s.name
                   HAVING COUNT(sp.id) >= 5
                   ORDER BY volatility_pct DESC
                   LIMIT ?""",
                (since_date.isoformat(), limit)
            )
            
            return [
                {
                    'symbol': row['symbol'],
                    'name': row['name'],
                    'avg_price': Decimal(str(row['avg_price'])),
                    'volatility_percent': Decimal(str(row['volatility_pct']))
                }
                for row in cursor.fetchall()
            ]
    
    async def get_stocks_for_analysis(
        self, 
        months_back: int = 6,
        min_data_points: int = 50
    ) -> List[Dict[str, Any]]:
        """Get stocks with sufficient data for analysis."""
        since_date = (datetime.now().date() - timedelta(days=months_back * 30))
        
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT s.*, COUNT(sp.id) as data_points, MAX(sp.date) as latest_date
                   FROM stocks s
                   JOIN stock_prices sp ON s.id = sp.stock_id
                   WHERE sp.date >= ?
                   GROUP BY s.id
                   HAVING COUNT(sp.id) >= ?
                   ORDER BY data_points DESC""",
                (since_date.isoformat(), min_data_points)
            )
            
            results = []
            for row in cursor.fetchall():
                stock = self._row_to_stock(row)
                results.append({
                    'stock': stock,
                    'data_points': row['data_points'],
                    'latest_date': datetime.fromisoformat(row['latest_date']).date()
                })
            
            return results

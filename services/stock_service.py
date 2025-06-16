# services/stock_service.py
from typing import List, Optional, Dict
import pandas as pd
from database.connection import db_manager
from models.stock import Stock, StockPrice, NewsItem, TechnicalIndicators

class StockService:
    @staticmethod
    def get_or_create_stock(symbol: str, name: str = None, exchange: str = None) -> int:
        """Get stock ID or create new stock entry."""
        # Try to get existing stock
        result = db_manager.execute_query(
            "SELECT id FROM stocks WHERE symbol = ?", (symbol,)
        )
        
        if result:
            return result[0]['id']
        
        # Create new stock
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO stocks (symbol, name, exchange) VALUES (?, ?, ?)",
                (symbol, name, exchange)
            )
            conn.commit()
            return cursor.lastrowid
    
    @staticmethod
    def get_stock_prices(stock_id: int, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get stock prices as DataFrame."""
        query = """
            SELECT date, open, high, low, close, volume, adjusted_close
            FROM stock_prices 
            WHERE stock_id = ?
        """
        params = [stock_id]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date ASC"
        
        with db_manager.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
    
    @staticmethod
    def bulk_insert_prices(prices: List[StockPrice]) -> None:
        """Bulk insert stock prices."""
        if not prices:
            return
        
        query = """
            INSERT OR REPLACE INTO stock_prices 
            (stock_id, date, open, high, low, close, volume, adjusted_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = [
            (p.stock_id, p.date, p.open, p.high, p.low, p.close, p.volume, p.adjusted_close)
            for p in prices
        ]
        
        db_manager.execute_many(query, params)
    
    @staticmethod
    def get_stock_info(stock_id: int) -> Optional[Dict]:
        """Get stock information."""
        result = db_manager.execute_query(
            """SELECT s.symbol, s.name, s.exchange, si.sector, si.industry, 
                      si.market_cap, si.beta, si.pe_ratio, si.dividend_yield
               FROM stocks s
               LEFT JOIN stock_info si ON s.id = si.stock_id
               WHERE s.id = ?""",
            (stock_id,)
        )
        
        return dict(result[0]) if result else None
    
    @staticmethod
    def update_stock_info(stock_id: int, **kwargs) -> None:
        """Update stock information."""
        fields = ['sector', 'industry', 'market_cap', 'beta', 'pe_ratio', 'dividend_yield']
        updates = {k: v for k, v in kwargs.items() if k in fields and v is not None}
        
        if not updates:
            return
        
        query = f"""
            INSERT OR REPLACE INTO stock_info 
            (stock_id, {', '.join(updates.keys())}, updated_at)
            VALUES (?, {', '.join(['?' for _ in updates])}, CURRENT_TIMESTAMP)
        """
        
        params = [stock_id] + list(updates.values())
        db_manager.execute_query(query, params)
    
    @staticmethod
    def get_all_stocks_with_data(months_back: int = 6) -> pd.DataFrame:
        """Get all stocks with price data and metadata."""
        query = """
            SELECT s.id as stock_id, s.symbol, s.name, s.exchange,
                   si.sector, si.industry, si.market_cap, si.beta,
                   sp.date, sp.open, sp.high, sp.low, sp.close, sp.volume
            FROM stocks s
            JOIN stock_prices sp ON s.id = sp.stock_id
            LEFT JOIN stock_info si ON s.id = si.stock_id
            WHERE sp.date >= date('now', '-{} months')
            ORDER BY s.symbol, sp.date
        """.format(months_back)
        
        with db_manager.get_connection() as conn:
            return pd.read_sql_query(query, conn, parse_dates=['date'])
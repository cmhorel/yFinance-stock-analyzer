# database_manager.py
"""
Centralized database manager for all database operations in the stock analysis application.
"""
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Centralized database manager for all application database operations."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_NAME
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def get_cash_query(self, date_str) -> str:
        with self.get_connection() as conn:
            cur = conn.execute( '''
            SELECT cash_balance FROM portfolio_state
            WHERE last_transaction_date <= ?
            ORDER BY last_transaction_date DESC LIMIT 1
            ''', (date_str,))
        row = cur.fetchone()
        cash = row[0] if row else 10000.0  # fallback to initial cash
        return cash
    
    def get_holdings_query(self) -> str:
        with self.get_connection() as conn:
            holdings_query = '''
                SELECT ph.symbol, ph.quantity
                FROM portfolio_holdings ph
                JOIN stocks s ON ph.stock_id = s.id
                WHERE ph.quantity > 0
            '''
            holdings = pd.read_sql_query(holdings_query, conn)
        return holdings

    # Stock and Stock Info Operations
    def get_or_create_stock_id(self, symbol: str) -> int:
        """Get stock ID, creating the stock record if it doesn't exist."""
        with self.get_connection() as conn:
            conn.execute('INSERT OR IGNORE INTO stocks (symbol) VALUES (?)', (symbol,))
            cursor = conn.execute('SELECT id FROM stocks WHERE symbol = ?', (symbol,))
            return cursor.fetchone()[0]
    
    def store_stock_info(self, stock_id: int, sector: str, industry: str) -> None:
        """Store stock sector and industry information."""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO stock_info (stock_id, sector, industry)
                VALUES (?, ?, ?)
            ''', (stock_id, sector, industry))
            conn.commit()
    
    def get_latest_stock_price_date(self, symbol: str) -> Optional[str]:
        """Get the latest date for stock price data."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT MAX(sp.date) 
                FROM stock_prices sp
                JOIN stocks s ON sp.stock_id = s.id
                WHERE s.symbol = ?
            ''', (symbol,))
            result = cursor.fetchone()
            return result[0] if result[0] else None
    
    # Stock Price Operations
    def store_stock_prices(self, stock_id: int, price_data: List[Dict[str, Any]]) -> None:
        """Store stock price data."""
        with self.get_connection() as conn:
            for data in price_data:
                try:
                    conn.execute('''
                        INSERT OR IGNORE INTO stock_prices
                        (stock_id, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        stock_id,
                        data['date'],
                        float(data['open']),
                        float(data['high']),
                        float(data['low']),
                        float(data['close']),
                        int(data['volume'])
                    ))
                except Exception as e:
                    logger.error(f"Error storing price data: {e}")
            conn.commit()
    
    def get_stock_data(self, months_back: int = 6) -> pd.DataFrame:
        """Get stock data for analysis."""
        cutoff_date = datetime.now() - timedelta(days=months_back * 30)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        query = '''
        SELECT st.id AS stock_id, st.symbol, sp.date, sp.close, sp.volume,
               si.industry, si.sector
        FROM stock_prices sp
        JOIN stocks st ON sp.stock_id = st.id
        LEFT JOIN stock_info si ON st.id = si.stock_id  
        WHERE sp.date >= ?
        ORDER BY st.symbol, sp.date
        '''
        
        with self.get_connection() as conn:
            try:
                df = pd.read_sql_query(query, conn, params=(cutoff_str,))
                df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception as e:
                logger.error(f"SQL query error: {e}")
                return pd.DataFrame()
    
    # News and Sentiment Operations
    def store_news_sentiments(self, stock_id: int, news_sentiments: List[Dict[str, Any]]) -> None:
        """Store news sentiment data."""
        with self.get_connection() as conn:
            for sentiment in news_sentiments:
                try:
                    # Handle datetime objects
                    pub_date = sentiment['publish_date']
                    if isinstance(pub_date, datetime):
                        pub_date = pub_date.strftime('%Y-%m-%d')
                    elif pub_date == 'N/A':
                        pub_date = datetime.now().strftime('%Y-%m-%d')
                    
                    conn.execute('''
                        INSERT OR IGNORE INTO stock_news 
                        (stock_id, date, title, summary, sentiment_score)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        stock_id, 
                        pub_date,
                        sentiment['title'][:500],  # Limit title length
                        sentiment['summary'][:1000],  # Limit summary length
                        sentiment['sentiment_score']
                    ))
                except Exception as e:
                    logger.error(f"Error storing news sentiment: {e}")
            
            conn.commit()
    
    def get_average_sentiment(self, stock_id: int, days_back: int = 7) -> float:
        """Get average sentiment for a stock over specified days."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        with self.get_connection() as conn:
            cursor = conn.execute(f'''
                SELECT AVG(sentiment_score) 
                FROM stock_news
                WHERE stock_id = {stock_id} AND date >= {cutoff_date}
            ''')
            
            result = cursor.fetchone()
            return result[0] if result[0] is not None else 0.0
    
    def get_sentiment_timeseries(self, stock_id: int, days_back: int = 60) -> pd.DataFrame:
        """Get daily average sentiment scores for a stock."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        with self.get_connection() as conn:
            query = '''
                SELECT date, AVG(sentiment_score) as avg_sentiment
                FROM stock_news
                WHERE stock_id = ? AND date >= ?
                GROUP BY date
                ORDER BY date ASC
            '''
            df = pd.read_sql_query(query, conn, params=(stock_id, cutoff_date))

        if df.empty:
            return pd.DataFrame({'date': [], 'avg_sentiment': []})
        
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def get_industry_stocks(self, industry: str, exclude_stock_id: int = None) -> List[int]:
        """Get all stock IDs in a specific industry."""
        with self.get_connection() as conn:
            if exclude_stock_id:
                cursor = conn.execute('''
                    SELECT stock_id FROM stock_info 
                    WHERE industry = ? AND stock_id != ?
                ''', (industry, exclude_stock_id))
            else:
                cursor = conn.execute('''
                    SELECT stock_id FROM stock_info 
                    WHERE industry = ?
                ''', (industry,))
            
            return [row[0] for row in cursor.fetchall()]
    
    # Portfolio Operations
    def get_portfolio_state(self) -> Tuple[Optional[Tuple], List[Tuple]]:
        """Get current portfolio state and holdings."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM portfolio_state ORDER BY id DESC LIMIT 1')
            portfolio = cursor.fetchone()
            cursor = conn.execute('''
                SELECT ph.*, s.symbol 
                FROM portfolio_holdings ph
                JOIN stocks s ON ph.stock_id = s.id
                WHERE ph.quantity > 0
            ''')
            holdings = cursor.fetchall()
            
            return portfolio, holdings
    
    def update_portfolio_cash(self, new_cash: float, transaction_date: str) -> None:
        """Update portfolio cash balance."""
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE portfolio_state 
                SET cash_balance = ?, last_transaction_date = ?
                WHERE id = (SELECT MAX(id) FROM portfolio_state)
            ''', (new_cash, transaction_date))
            conn.commit()
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update total portfolio value."""
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE portfolio_state 
                SET total_portfolio_value = ?
                WHERE id = (SELECT MAX(id) FROM portfolio_state)
            ''', (new_value,))
            conn.commit()
    
    def record_transaction(self, stock_id: int, symbol: str, transaction_type: str, 
                          quantity: int, price_per_share: float, total_amount: float, 
                          brokerage_fee: float, transaction_date: str) -> None:
        """Record a portfolio transaction."""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO portfolio_transactions 
                (stock_id, symbol, transaction_type, quantity, price_per_share, total_amount, brokerage_fee, transaction_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (stock_id, symbol, transaction_type, quantity, price_per_share, total_amount, brokerage_fee, transaction_date))
            conn.commit()
    
    def update_or_create_holding(self, stock_id: int, symbol: str, quantity: int, 
                                avg_cost: float, total_cost: float) -> None:
        """Update existing holding or create new one."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM portfolio_holdings WHERE stock_id = ?', (stock_id,))
            existing_holding = cursor.fetchone()
            
            if existing_holding:
                # Update existing holding
                old_quantity = existing_holding[2]
                old_total_cost = existing_holding[4]
                new_quantity = old_quantity + quantity
                new_total_cost = old_total_cost + total_cost
                new_avg_cost = new_total_cost / new_quantity
                
                conn.execute('''
                    UPDATE portfolio_holdings 
                    SET quantity = ?, avg_cost_per_share = ?, total_cost = ?
                    WHERE stock_id = ?
                ''', (new_quantity, new_avg_cost, new_total_cost, stock_id))
            else:
                # Insert new holding
                conn.execute('''
                    INSERT INTO portfolio_holdings (stock_id, symbol, quantity, avg_cost_per_share, total_cost)
                    VALUES (?, ?, ?, ?, ?)
                ''', (stock_id, symbol, quantity, avg_cost, total_cost))
            
            conn.commit()
    
    def reduce_or_remove_holding(self, stock_id: int, quantity_to_sell: int) -> None:
        """Reduce holding quantity or remove if selling all."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM portfolio_holdings WHERE stock_id = ?', (stock_id,))
            holding = cursor.fetchone()
            
            if holding:
                old_quantity = holding[2]
                old_total_cost = holding[4]
                new_quantity = old_quantity - quantity_to_sell
                
                if new_quantity <= 0:
                    # Remove holding entirely
                    conn.execute('DELETE FROM portfolio_holdings WHERE stock_id = ?', (stock_id,))
                else:
                    # Reduce holding proportionally
                    cost_per_share = old_total_cost / old_quantity
                    new_total_cost = new_quantity * cost_per_share
                    conn.execute('''
                        UPDATE portfolio_holdings 
                        SET quantity = ?, total_cost = ?
                        WHERE stock_id = ?
                    ''', (new_quantity, new_total_cost, stock_id))
                
                conn.commit()
    
    def get_portfolio_transactions(self) -> pd.DataFrame:
        """Get all portfolio transactions."""
        with self.get_connection() as conn:
            return pd.read_sql_query(
                'SELECT * FROM portfolio_transactions ORDER BY transaction_date ASC', conn)

# Create global instance
db_manager = DatabaseManager()
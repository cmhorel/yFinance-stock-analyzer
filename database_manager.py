# news_analyzer/database_manager.py
"""
Database operations for news and sentiment data.
"""
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import app.appconfig as appconfig

logger = logging.getLogger(__name__)

class NewsDatabase:
    """Handles all database operations for news and sentiment data."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or appconfig.DB_NAME
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def store_stock_info(self, stock_id: int, sector: str, industry: str) -> None:
        """Store stock sector and industry information."""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO stock_info (stock_id, sector, industry)
                VALUES (?, ?, ?)
            ''', (stock_id, sector, industry))
            conn.commit()
    
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
        print(f'''
                SELECT AVG(sentiment_score) 
                FROM stock_news
                WHERE stock_id = {stock_id} AND date >= {cutoff_date}
            ''')
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
            query = f'''
                SELECT date, AVG(sentiment_score) as avg_sentiment
                FROM stock_news
                WHERE stock_id = {stock_id} AND date >= {cutoff_date}
                GROUP BY date
                ORDER BY date ASC
            '''
            df = pd.read_sql_query(query, conn)

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
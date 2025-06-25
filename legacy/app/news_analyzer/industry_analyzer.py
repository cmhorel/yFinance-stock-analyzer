# news_analyzer/industry_analyzer.py
"""
Industry-level analysis and comparisons.
"""
import logging
import pandas as pd
from typing import Optional
from app.database_manager import DatabaseManager
logger = logging.getLogger(__name__)

class IndustryAnalyzer:
    """Handles industry-level analysis and comparisons."""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def get_industry_average_momentum(self, industry: str, exclude_stock_id: int, 
                                    df_all: pd.DataFrame) -> float:
        """
        Calculate average momentum for stocks in the same industry.
        
        Args:
            industry: Industry name
            exclude_stock_id: Stock ID to exclude from calculation
            df_all: DataFrame with all stock data
            
        Returns:
            Average momentum or 0.0 if insufficient data
        """
        try:
            # Get stocks in the same industry
            industry_stock_ids = self.db.get_industry_stocks(industry, exclude_stock_id)
            
            if not industry_stock_ids:
                return 0.0
            
            momenta = []
            for stock_id in industry_stock_ids:
                df_stock = df_all[df_all['stock_id'] == stock_id]
                
                if len(df_stock) < 8:  # Need at least 8 points for 7-day momentum
                    continue
                
                # Calculate momentum (current close - close 7 days ago)
                df_stock = df_stock.sort_values('date')
                momentum = df_stock['close'].iloc[-1] - df_stock['close'].iloc[-8]
                
                if pd.notna(momentum):
                    momenta.append(momentum)
            
            return sum(momenta) / len(momenta) if momenta else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating industry momentum for {industry}: {e}")
            return 0.0
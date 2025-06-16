# analysis/technical_indicators.py
import pandas as pd
import numpy as np
from typing import Dict, Optional
from services.stock_service import StockService
from database.connection import db_manager

class TechnicalAnalyzer:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages."""
        return {
            'ma_20': prices.rolling(window=20, min_periods=1).mean(),
            'ma_50': prices.rolling(window=50, min_periods=1).mean(),
            'ma_200': prices.rolling(window=200, min_periods=1).mean()
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        
        return {
            'bollinger_upper': ma + (std * std_dev),
            'bollinger_lower': ma - (std * std_dev)
        }
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return {
            'macd': macd,
            'macd_signal': macd_signal
        }
    
    @staticmethod
    def calculate_all_indicators(stock_id: int) -> None:
        """Calculate and cache all technical indicators for a stock."""
        df = StockService.get_stock_prices(stock_id)
        
        if len(df) < 200:  # Need sufficient data
            return
        
        # Calculate all indicators
        rsi = TechnicalAnalyzer.calculate_rsi(df['close'])
        mas = TechnicalAnalyzer.calculate_moving_averages(df['close'])
        bollinger = TechnicalAnalyzer.calculate_bollinger_bands(df['close'])
        macd = TechnicalAnalyzer.calculate_macd(df['close'])
        volume_sma = df['volume'].rolling(window=20, min_periods=1).mean()
        
        # Prepare data for insertion
        indicators_data = []
        for i, row in df.iterrows():
            indicators_data.append((
                stock_id,
                row['date'].strftime('%Y-%m-%d'),
                rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else None,
                mas['ma_20'].iloc[i] if not pd.isna(mas['ma_20'].iloc[i]) else None,
                mas['ma_50'].iloc[i] if not pd.isna(mas['ma_50'].iloc[i]) else None,
                mas['ma_200'].iloc[i] if not pd.isna(mas['ma_200'].iloc[i]) else None,
                bollinger['bollinger_upper'].iloc[i] if not pd.isna(bollinger['bollinger_upper'].iloc[i]) else None,
                bollinger['bollinger_lower'].iloc[i] if not pd.isna(bollinger['bollinger_lower'].iloc[i]) else None,
                macd['macd'].iloc[i] if not pd.isna(macd['macd'].iloc[i]) else None,
                macd['macd_signal'].iloc[i] if not pd.isna(macd['macd_signal'].iloc[i]) else None,
                volume_sma.iloc[i] if not pd.isna(volume_sma.iloc[i]) else None
            ))
        
        # Bulk insert
        query = """
            INSERT OR REPLACE INTO technical_indicators 
            (stock_id, date, rsi_14, ma_20, ma_50, ma_200, bollinger_upper, bollinger_lower, 
             macd, macd_signal, volume_sma_20)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        db_manager.execute_many(query, indicators_data)
    
    @staticmethod
    def get_latest_indicators(stock_id: int) -> Optional[Dict]:
        """Get the latest technical indicators for a stock."""
        result = db_manager.execute_query(
            """SELECT * FROM technical_indicators 
               WHERE stock_id = ? 
               ORDER BY date DESC 
               LIMIT 1""",
            (stock_id,)
        )
        
        return dict(result[0]) if result else None
"""Technical analysis service for calculating indicators and signals."""
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..entities.stock import StockPrice
from ..entities.analysis import TechnicalIndicators, Signal, SignalType, SignalStrength
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.config import get_settings
from shared.logging import get_logger


class TechnicalAnalysisService:
    """Service for technical analysis calculations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
    
    def calculate_rsi(self, prices: List[StockPrice], period: int = None) -> List[Decimal]:
        """Calculate RSI (Relative Strength Index) for a series of prices."""
        if period is None:
            period = self.settings.analysis.rsi_period
        
        period = int(period)  # Ensure it's an int
        
        if len(prices) < period + 1:
            return [Decimal('50')] * len(prices)  # Neutral RSI for insufficient data
        
        # Convert to pandas series for calculation
        close_prices = pd.Series([float(price.close_price) for price in prices])
        
        delta = close_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Fill NaN values with neutral RSI
        rsi = rsi.fillna(50)
        
        return [Decimal(str(val)) for val in rsi.tolist()]
    
    def calculate_moving_averages(
        self, 
        prices: List[StockPrice], 
        periods: List[int] = None
    ) -> Dict[int, List[Decimal]]:
        """Calculate moving averages for specified periods."""
        if periods is None:
            periods = [
                self.settings.analysis.ma_short_period,
                self.settings.analysis.ma_long_period
            ]
        
        periods = [int(p) for p in periods]  # Ensure they're ints
        
        close_prices = pd.Series([float(price.close_price) for price in prices])
        result = {}
        
        for period in periods:
            ma = close_prices.rolling(window=period).mean()
            ma = ma.bfill()  # Forward fill for initial values
            result[period] = [Decimal(str(val)) for val in ma.tolist()]
        
        return result
    
    def calculate_volatility(
        self, 
        prices: List[StockPrice], 
        period: int = None
    ) -> List[Decimal]:
        """Calculate rolling volatility (annualized standard deviation of returns)."""
        if period is None:
            period = self.settings.analysis.volatility_period
        
        period = int(period)  # Ensure it's an int
        
        if len(prices) < period:
            return [Decimal('0.2')] * len(prices)  # Default volatility
        
        close_prices = pd.Series([float(price.close_price) for price in prices])
        returns = close_prices.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        volatility = volatility.fillna(0.2)  # Default volatility for NaN values
        
        return [Decimal(str(val)) for val in volatility.tolist()]
    
    def calculate_momentum(
        self, 
        prices: List[StockPrice], 
        period: int = None
    ) -> List[Decimal]:
        """Calculate price momentum over specified period."""
        if period is None:
            period = self.settings.analysis.momentum_period
        
        period = int(period)  # Ensure it's an int
        
        if len(prices) < period + 1:
            return [Decimal('0')] * len(prices)
        
        close_prices = pd.Series([float(price.close_price) for price in prices])
        momentum = close_prices.pct_change(periods=period)
        momentum = momentum.fillna(0)
        
        return [Decimal(str(val)) for val in momentum.tolist()]
    
    def calculate_volume_ratio(
        self, 
        prices: List[StockPrice], 
        period: int = None
    ) -> List[Decimal]:
        """Calculate volume ratio (current volume vs average volume)."""
        if period is None:
            period = self.settings.analysis.volume_period
        
        period = int(period)  # Ensure it's an int
        
        if len(prices) < period:
            return [Decimal('1')] * len(prices)
        
        volumes = pd.Series([float(price.volume) for price in prices])
        avg_volume = volumes.rolling(window=period).mean()
        volume_ratio = volumes / avg_volume
        volume_ratio = volume_ratio.fillna(1)
        
        return [Decimal(str(val)) for val in volume_ratio.tolist()]
    
    def create_technical_indicators(
        self, 
        stock_id: int, 
        prices: List[StockPrice]
    ) -> List[TechnicalIndicators]:
        """Create technical indicators for all price data points."""
        if not prices:
            return []
        
        # Calculate all indicators
        rsi_values = self.calculate_rsi(prices)
        ma_values = self.calculate_moving_averages(prices)
        volatility_values = self.calculate_volatility(prices)
        momentum_7d = self.calculate_momentum(prices, 7)
        momentum_30d = self.calculate_momentum(prices, 30)
        volume_ratios = self.calculate_volume_ratio(prices)
        
        indicators = []
        for i, price in enumerate(prices):
            indicator = TechnicalIndicators(
                stock_id=stock_id,
                date=price.date,
                ma_20=ma_values.get(20, [None] * len(prices))[i],
                ma_50=ma_values.get(50, [None] * len(prices))[i],
                rsi=rsi_values[i],
                volatility=volatility_values[i],
                momentum_7d=momentum_7d[i],
                momentum_30d=momentum_30d[i],
                volume_ratio=volume_ratios[i],
                price_range=price.high_price - price.low_price,
                daily_return=price.daily_return
            )
            indicators.append(indicator)
        
        return indicators
    
    def generate_buy_signal(
        self, 
        indicators: TechnicalIndicators,
        stock_id: int,
        symbol: str,
        current_price: Decimal,
        sentiment_score: Optional[Decimal] = None
    ) -> Optional[Signal]:
        """Generate a buy signal based on technical indicators."""
        conditions = []
        reasons = []
        
        # Price above 20-day MA
        if indicators.ma_20 and current_price > indicators.ma_20:
            conditions.append(True)
            reasons.append("Price above 20-day MA")
        else:
            conditions.append(False)
        
        # Uptrend (20-day MA > 50-day MA)
        if indicators.is_uptrend:
            conditions.append(True)
            reasons.append("Uptrend confirmed")
        else:
            conditions.append(False)
        
        # RSI oversold
        if indicators.is_oversold:
            conditions.append(True)
            reasons.append("RSI oversold")
        else:
            conditions.append(False)
        
        # Positive momentum
        if indicators.momentum_7d and indicators.momentum_7d > self.settings.analysis.momentum_threshold:
            conditions.append(True)
            reasons.append("Positive momentum")
        else:
            conditions.append(False)
        
        # Volume surge
        if indicators.volume_ratio and indicators.volume_ratio > self.settings.analysis.volume_surge_threshold:
            conditions.append(True)
            reasons.append("Volume surge")
        else:
            conditions.append(False)
        
        # Reasonable volatility
        if indicators.volatility and indicators.volatility <= self.settings.analysis.volatility_low_threshold:
            conditions.append(True)
            reasons.append("Low volatility")
        else:
            conditions.append(False)
        
        # Positive sentiment (if available)
        if sentiment_score and sentiment_score > self.settings.news.sentiment_positive_threshold:
            conditions.append(True)
            reasons.append("Positive sentiment")
        
        # Calculate score and determine signal strength
        score = sum(conditions)
        total_conditions = len(conditions)
        
        if score >= self.settings.analysis.min_buy_conditions:
            confidence = Decimal(str(score / total_conditions))
            
            if score >= total_conditions * 0.8:
                strength = SignalStrength.STRONG
            elif score >= total_conditions * 0.6:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            return Signal(
                id=None,
                stock_id=stock_id,
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                score=score,
                confidence=confidence,
                price=current_price,
                generated_date=datetime.now(),
                reasons="; ".join(reasons),
                metadata={
                    "total_conditions": total_conditions,
                    "met_conditions": score,
                    "rsi": float(indicators.rsi) if indicators.rsi else None,
                    "volatility": float(indicators.volatility) if indicators.volatility else None,
                    "momentum_7d": float(indicators.momentum_7d) if indicators.momentum_7d else None
                }
            )
        
        return None
    
    def generate_sell_signal(
        self, 
        indicators: TechnicalIndicators,
        stock_id: int,
        symbol: str,
        current_price: Decimal,
        sentiment_score: Optional[Decimal] = None
    ) -> Optional[Signal]:
        """Generate a sell signal based on technical indicators."""
        conditions = []
        reasons = []
        
        # Price below 20-day MA
        if indicators.ma_20 and current_price < indicators.ma_20:
            conditions.append(True)
            reasons.append("Price below 20-day MA")
        else:
            conditions.append(False)
        
        # Downtrend (20-day MA < 50-day MA)
        if indicators.is_downtrend:
            conditions.append(True)
            reasons.append("Downtrend confirmed")
        else:
            conditions.append(False)
        
        # RSI overbought
        if indicators.is_overbought:
            conditions.append(True)
            reasons.append("RSI overbought")
        else:
            conditions.append(False)
        
        # Negative momentum
        if indicators.momentum_7d and indicators.momentum_7d < -self.settings.analysis.momentum_threshold:
            conditions.append(True)
            reasons.append("Negative momentum")
        else:
            conditions.append(False)
        
        # High volatility
        if indicators.volatility and indicators.volatility > self.settings.analysis.volatility_high_threshold:
            conditions.append(True)
            reasons.append("High volatility")
        else:
            conditions.append(False)
        
        # Negative sentiment (if available)
        if sentiment_score and sentiment_score < self.settings.news.sentiment_negative_threshold:
            conditions.append(True)
            reasons.append("Negative sentiment")
        
        # Calculate score and determine signal strength
        score = sum(conditions)
        total_conditions = len(conditions)
        
        if score >= self.settings.analysis.min_sell_conditions:
            confidence = Decimal(str(score / total_conditions))
            
            if score >= total_conditions * 0.8:
                strength = SignalStrength.STRONG
            elif score >= total_conditions * 0.6:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            return Signal(
                id=None,
                stock_id=stock_id,
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                score=score,
                confidence=confidence,
                price=current_price,
                generated_date=datetime.now(),
                reasons="; ".join(reasons),
                metadata={
                    "total_conditions": total_conditions,
                    "met_conditions": score,
                    "rsi": float(indicators.rsi) if indicators.rsi else None,
                    "volatility": float(indicators.volatility) if indicators.volatility else None,
                    "momentum_7d": float(indicators.momentum_7d) if indicators.momentum_7d else None
                }
            )
        
        return None

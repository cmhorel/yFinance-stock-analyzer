# analysis/trading_strategy.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from services.stock_service import StockService
from analysis.technical_indicators import TechnicalAnalyzer
from analysis.sentiment_analyzer import SentimentAnalyzer

@dataclass
class TradingSignal:
    stock_id: int
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    score: float
    reasons: List[str]
    sector: Optional[str] = None
    industry: Optional[str] = None
    current_price: Optional[float] = None

class AdvancedTradingStrategy:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_stock(self, stock_id: int, df_all: pd.DataFrame) -> Optional[TradingSignal]:
        """Comprehensive stock analysis using multiple factors."""
        stock_info = StockService.get_stock_info(stock_id)
        if not stock_info:
            return None
        
        symbol = stock_info['symbol']
        sector = stock_info.get('sector', 'Unknown')
        industry = stock_info.get('industry', 'Unknown')
        
        # Get stock data
        stock_data = df_all[df_all['stock_id'] == stock_id].copy()
        if len(stock_data) < 50:  # Need sufficient data
            return None
        
        stock_data = stock_data.sort_values('date')
        
        # Get technical indicators
        indicators = TechnicalAnalyzer.get_latest_indicators(stock_id)
        if not indicators:
            return None
        
        # Calculate additional metrics
        current_price = stock_data['close'].iloc[-1]
        price_change_1d = (current_price - stock_data['close'].iloc[-2]) / stock_data['close'].iloc[-2]
        price_change_5d = (current_price - stock_data['close'].iloc[-6]) / stock_data['close'].iloc[-6] if len(stock_data) >= 6 else 0
        price_change_20d = (current_price - stock_data['close'].iloc[-21]) / stock_data['close'].iloc[-21] if len(stock_data) >= 21 else 0
        
        # Volume analysis
        avg_volume_20d = stock_data['volume'].tail(20).mean()
        current_volume = stock_data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1
        
        # Volatility (20-day rolling standard deviation of returns)
        returns = stock_data['close'].pct_change().dropna()
        volatility = returns.tail(20).std() * np.sqrt(252)  # Annualized
        
        # Sentiment analysis
        avg_sentiment = self.sentiment_analyzer.get_average_sentiment(stock_id, days_back=7)
        sentiment_trend = self.sentiment_analyzer.get_sentiment_trend(stock_id, days_back=14)
        
        # Industry comparison
        industry_performance = self._get_industry_performance(industry, stock_id, df_all)
        
        # Generate trading signal
        return self._generate_signal(
            stock_id=stock_id,
            symbol=symbol,
            sector=sector,
            industry=industry,
            current_price=current_price,
            indicators=indicators,
            price_changes={
                '1d': price_change_1d,
                '5d': price_change_5d,
                '20d': price_change_20d
            },
            volume_ratio=volume_ratio,
            volatility=volatility,
            sentiment_score=avg_sentiment,
            sentiment_trend=sentiment_trend,
            industry_performance=industry_performance
        )
    
    def _generate_signal(self, **kwargs) -> TradingSignal:
        """Generate trading signal based on multiple factors."""
        stock_id = kwargs['stock_id']
        symbol = kwargs['symbol']
        sector = kwargs['sector']
        industry = kwargs['industry']
        current_price = kwargs['current_price']
        indicators = kwargs['indicators']
        price_changes = kwargs['price_changes']
        volume_ratio = kwargs['volume_ratio']
        volatility = kwargs['volatility']
        sentiment_score = kwargs['sentiment_score']
        sentiment_trend = kwargs['sentiment_trend']
        industry_performance = kwargs['industry_performance']
        
        buy_score = 0
        sell_score = 0
        reasons = []
        
        # Technical Analysis Factors (40% weight)
        
        # RSI analysis
        rsi = indicators.get('rsi_14', 50)
        if rsi < 30:
            buy_score += 2
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            sell_score += 2
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif 30 <= rsi <= 45:
            buy_score += 1
            reasons.append(f"RSI favorable for buying ({rsi:.1f})")
        elif 55 <= rsi <= 70:
            sell_score += 1
            reasons.append(f"RSI suggests selling pressure ({rsi:.1f})")
        
        # Moving Average analysis
        ma_20 = indicators.get('ma_20')
        ma_50 = indicators.get('ma_50')
        ma_200 = indicators.get('ma_200')
        
        if ma_20 and ma_50 and ma_200:
            if current_price > ma_20 > ma_50 > ma_200:
                buy_score += 3
                reasons.append("Strong uptrend (price > MA20 > MA50 > MA200)")
            elif current_price > ma_20 > ma_50:
                buy_score += 2
                reasons.append("Uptrend (price > MA20 > MA50)")
            elif current_price < ma_20 < ma_50 < ma_200:
                sell_score += 3
                reasons.append("Strong downtrend (price < MA20 < MA50 < MA200)")
            elif current_price < ma_20 < ma_50:
                sell_score += 2
                reasons.append("Downtrend (price < MA20 < MA50)")
        
        # MACD analysis
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        
        if macd and macd_signal:
            if macd > macd_signal and macd > 0:
                buy_score += 2
                reasons.append("MACD bullish crossover above zero")
            elif macd < macd_signal and macd < 0:
                sell_score += 2
                reasons.append("MACD bearish crossover below zero")
        
        # Bollinger Bands analysis
        bb_upper = indicators.get('bollinger_upper')
        bb_lower = indicators.get('bollinger_lower')
        
        if bb_upper and bb_lower:
            if current_price <= bb_lower:
                buy_score += 1
                reasons.append("Price at lower Bollinger Band")
            elif current_price >= bb_upper:
                sell_score += 1
                reasons.append("Price at upper Bollinger Band")
        
        # Momentum Analysis (25% weight)
        
        # Price momentum
        if price_changes['5d'] > 0.05:  # 5% gain in 5 days
            buy_score += 2
            reasons.append(f"Strong 5-day momentum (+{price_changes['5d']:.1%})")
        elif price_changes['5d'] < -0.05:  # 5% loss in 5 days
            sell_score += 2
            reasons.append(f"Weak 5-day momentum ({price_changes['5d']:.1%})")
        
        # Volume confirmation
        if volume_ratio > 1.5:  # 50% above average volume
            if price_changes['1d'] > 0:
                buy_score += 1
                reasons.append("High volume with price increase")
            else:
                sell_score += 1
                reasons.append("High volume with price decrease")
        
        # Sentiment Analysis (20% weight)
        
        if sentiment_score > 0.2:
            buy_score += 2
            reasons.append(f"Positive sentiment ({sentiment_score:.2f})")
        elif sentiment_score < -0.2:
            sell_score += 2
            reasons.append(f"Negative sentiment ({sentiment_score:.2f})")
        
        if sentiment_trend > 0.1:
            buy_score += 1
            reasons.append("Improving sentiment trend")
        elif sentiment_trend < -0.1:
            sell_score += 1
            reasons.append("Deteriorating sentiment trend")
        
        # Industry/Market Analysis (15% weight)
        
        if industry_performance > 0.02:  # Industry outperforming by 2%
            buy_score += 1
            reasons.append("Industry outperforming market")
        elif industry_performance < -0.02:  # Industry underperforming by 2%
            sell_score += 1
            reasons.append("Industry underperforming market")
        
        # Risk Management
        
        # High volatility penalty
        if volatility > 0.4:  # 40% annualized volatility
            buy_score -= 1
            sell_score += 1
            reasons.append(f"High volatility risk ({volatility:.1%})")
        
        # Determine signal
        net_score = buy_score - sell_score
        
        if net_score >= 3:
            signal_type = 'BUY'
            confidence = min(0.9, (net_score / 10) + 0.5)
            score = buy_score
        elif net_score <= -3:
            signal_type = 'SELL'
            confidence = min(0.9, (abs(net_score) / 10) + 0.5)
            score = sell_score
        else:
            signal_type = 'HOLD'
            confidence = 0.3
            score = 0
        
        return TradingSignal(
            stock_id=stock_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            score=score,
            reasons=reasons,
            sector=sector,
            industry=industry,
            current_price=current_price
        )
    
    def _get_industry_performance(self, industry: str, exclude_stock_id: int, df_all: pd.DataFrame) -> float:
        """Calculate industry performance relative to market."""
        if industry == 'Unknown':
            return 0.0
        
        # Get industry stocks (excluding current stock)
        industry_stocks = df_all[
            (df_all['industry'] == industry) & 
            (df_all['stock_id'] != exclude_stock_id)
        ]['stock_id'].unique()
        
        if len(industry_stocks) < 3:  # Need at least 3 stocks for comparison
            return 0.0
        
        # Calculate 20-day performance for
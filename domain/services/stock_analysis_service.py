"""Stock analysis service that orchestrates the analysis workflow."""
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime

from ..entities.stock import Stock, StockPrice
from ..entities.analysis import AnalysisResult, TechnicalIndicators, Signal
from ..entities.news import SentimentScore
from .technical_analysis_service import TechnicalAnalysisService


class StockAnalysisService:
    """Service for orchestrating stock analysis workflow."""
    
    def __init__(self):
        self.technical_service = TechnicalAnalysisService()
    
    def analyze_stock(
        self, 
        stock: Stock, 
        prices: List[StockPrice],
        sentiment_scores: List[SentimentScore] = None
    ) -> Optional[AnalysisResult]:
        """Perform complete analysis for a stock."""
        if not prices or len(prices) < 50:  # Minimum data requirement
            return None
        
        # Calculate technical indicators
        indicators_list = self.technical_service.create_technical_indicators(stock.id, prices)
        if not indicators_list:
            return None
        
        # Get latest indicators
        latest_indicators = indicators_list[-1]
        latest_price = prices[-1]
        
        # Calculate average sentiment
        avg_sentiment = None
        sentiment_count = 0
        if sentiment_scores:
            total_sentiment = sum(score.score for score in sentiment_scores)
            sentiment_count = len(sentiment_scores)
            avg_sentiment = total_sentiment / sentiment_count if sentiment_count > 0 else None
        
        # Generate signals
        buy_signal = self.technical_service.generate_buy_signal(
            latest_indicators,
            stock.id,
            stock.symbol,
            latest_price.close_price,
            avg_sentiment
        )
        
        sell_signal = self.technical_service.generate_sell_signal(
            latest_indicators,
            stock.id,
            stock.symbol,
            latest_price.close_price,
            avg_sentiment
        )
        
        # Calculate risk score (simple volatility-based)
        risk_score = None
        if latest_indicators.volatility:
            # Scale volatility to 0-10 risk score
            vol = float(latest_indicators.volatility)
            risk_score = min(Decimal('10'), Decimal(str(vol * 20)))
        
        # Create analysis result
        result = AnalysisResult(
            id=None,
            stock_id=stock.id,
            symbol=stock.symbol,
            analysis_date=datetime.now(),
            indicators=latest_indicators,
            buy_signal=buy_signal,
            sell_signal=sell_signal,
            avg_sentiment=avg_sentiment,
            sentiment_count=sentiment_count,
            risk_score=risk_score,
            metadata={
                "price_count": len(prices),
                "latest_price": float(latest_price.close_price),
                "analysis_version": "1.0"
            }
        )
        
        return result
    
    def get_recommendations(
        self, 
        analysis_results: List[AnalysisResult]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get buy/sell recommendations from analysis results."""
        buy_recommendations = []
        sell_recommendations = []
        
        for result in analysis_results:
            if result.has_buy_signal:
                buy_recommendations.append({
                    "symbol": result.symbol,
                    "score": result.buy_signal.score,
                    "confidence": float(result.buy_signal.confidence),
                    "strength": result.buy_signal.strength.value,
                    "price": float(result.buy_signal.price),
                    "sentiment": float(result.avg_sentiment) if result.avg_sentiment else 0.0,
                    "risk_score": float(result.risk_score) if result.risk_score else 5.0,
                    "reasons": result.buy_signal.reasons
                })
            
            if result.has_sell_signal:
                sell_recommendations.append({
                    "symbol": result.symbol,
                    "score": result.sell_signal.score,
                    "confidence": float(result.sell_signal.confidence),
                    "strength": result.sell_signal.strength.value,
                    "price": float(result.sell_signal.price),
                    "sentiment": float(result.avg_sentiment) if result.avg_sentiment else 0.0,
                    "risk_score": float(result.risk_score) if result.risk_score else 5.0,
                    "reasons": result.sell_signal.reasons
                })
        
        # Sort by score and confidence
        buy_recommendations.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
        sell_recommendations.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
        
        return {
            "buy": buy_recommendations,
            "sell": sell_recommendations
        }

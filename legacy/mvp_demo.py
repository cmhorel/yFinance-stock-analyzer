"""
MVP Demonstration of the Refactored Stock Analyzer Architecture

This script demonstrates the new clean architecture with:
- Domain entities with business logic
- Domain services for technical analysis
- Configuration management
- Structured logging
- Exception handling
"""

import sys
import os
from datetime import datetime, date
from decimal import Decimal

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import the new architecture components
from shared.config import get_settings, setup_logging
from shared.logging import get_logger
from domain.entities.stock import Stock, StockPrice
from domain.entities.analysis import TechnicalIndicators
from domain.services.technical_analysis_service import TechnicalAnalysisService
from domain.services.stock_analysis_service import StockAnalysisService


def main():
    """Demonstrate the MVP refactored architecture."""
    
    # Initialize configuration and logging
    settings = get_settings()
    setup_logging(
        level=settings.logging.log_level,
        format_string=settings.logging.format
    )
    
    logger = get_logger(__name__)
    logger.info("Starting MVP demonstration of refactored stock analyzer")
    
    try:
        # Create sample stock entity
        stock = Stock(
            id=1,
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ"
        )
        logger.info(f"Created stock entity: {stock.symbol}")
        
        # Create sample price data
        prices = []
        base_price = Decimal('150.00')
        for i in range(60):  # 60 days of data
            price_change = Decimal(str((i % 10 - 5) * 0.5))  # Simple price variation
            current_price = base_price + price_change
            
            price = StockPrice(
                id=i + 1,
                stock_id=stock.id,
                date=datetime.now().date(),
                open_price=current_price - Decimal('0.50'),
                high_price=current_price + Decimal('1.00'),
                low_price=current_price - Decimal('1.00'),
                close_price=current_price,
                volume=1000000 + (i * 10000)
            )
            prices.append(price)
        
        logger.info(f"Generated {len(prices)} sample price records")
        
        # Demonstrate technical analysis service
        technical_service = TechnicalAnalysisService()
        logger.info("Initialized technical analysis service")
        
        # Calculate technical indicators
        indicators = technical_service.create_technical_indicators(stock.id, prices)
        logger.info(f"Calculated technical indicators for {len(indicators)} data points")
        
        if indicators:
            latest = indicators[-1]
            logger.info(f"Latest RSI: {latest.rsi}")
            logger.info(f"Latest volatility: {latest.volatility}")
            logger.info(f"Latest momentum: {latest.momentum_7d}")
        
        # Demonstrate stock analysis service
        analysis_service = StockAnalysisService()
        logger.info("Initialized stock analysis service")
        
        # Perform complete analysis
        result = analysis_service.analyze_stock(stock, prices)
        
        if result:
            logger.info(f"Analysis completed for {result.symbol}")
            logger.info(f"Risk score: {result.risk_score}")
            
            if result.has_buy_signal:
                logger.info(f"BUY SIGNAL: {result.buy_signal.strength.value} - {result.buy_signal.reasons}")
            
            if result.has_sell_signal:
                logger.info(f"SELL SIGNAL: {result.sell_signal.strength.value} - {result.sell_signal.reasons}")
            
            # Get recommendations
            recommendations = analysis_service.get_recommendations([result])
            
            if recommendations["buy"]:
                logger.info("Buy recommendations:")
                for rec in recommendations["buy"]:
                    logger.info(f"  {rec['symbol']}: Score {rec['score']}, Confidence {rec['confidence']:.2f}")
            
            if recommendations["sell"]:
                logger.info("Sell recommendations:")
                for rec in recommendations["sell"]:
                    logger.info(f"  {rec['symbol']}: Score {rec['score']}, Confidence {rec['confidence']:.2f}")
        
        # Demonstrate configuration system
        logger.info("Configuration demonstration:")
        logger.info(f"  RSI Period: {settings.analysis.rsi_period}")
        logger.info(f"  MA Short Period: {settings.analysis.ma_short_period}")
        logger.info(f"  Volatility Threshold: {settings.analysis.volatility_low_threshold}")
        logger.info(f"  Database Path: {settings.database.path}")
        
        logger.info("MVP demonstration completed successfully!")
        
        print("\n" + "="*60)
        print("🎉 MVP REFACTORING DEMONSTRATION COMPLETE!")
        print("="*60)
        print("\n✅ Successfully demonstrated:")
        print("  • Domain entities with validation")
        print("  • Technical analysis service")
        print("  • Stock analysis orchestration")
        print("  • Configuration management")
        print("  • Structured logging")
        print("  • Clean architecture separation")
        print("\n🏗️ Architecture Benefits:")
        print("  • Type-safe domain objects")
        print("  • Configurable business logic")
        print("  • Testable service layer")
        print("  • Structured error handling")
        print("  • Environment-based configuration")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()

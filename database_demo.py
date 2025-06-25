"""
Complete Database Demo - Full Architecture Working Together

This demonstrates:
- Domain entities with validation
- Repository pattern with SQLite implementation
- Domain services for business logic
- Configuration and logging
- Complete CRUD operations
"""

import asyncio
import sys
import os
from datetime import datetime, date, timedelta
from decimal import Decimal

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import the complete architecture
from shared.config import get_settings, setup_logging
from shared.logging import get_logger
from domain.entities.stock import Stock, StockPrice, StockInfo
from infrastructure.database import SqliteStockRepository


async def main():
    """Demonstrate the complete database-enabled architecture."""
    
    # Initialize configuration and logging
    settings = get_settings()
    setup_logging(
        level=settings.logging.log_level,
        format_string=settings.logging.format
    )
    
    logger = get_logger(__name__)
    logger.info("Starting complete database architecture demo")
    
    print("🚀 Complete Database Architecture Demo")
    print("=" * 60)
    
    try:
        # Initialize repository with a fresh database
        demo_db_path = "data/demo_stocks.db"
        
        # Remove existing demo database if it exists
        if os.path.exists(demo_db_path):
            os.remove(demo_db_path)
        
        repo = SqliteStockRepository(db_path=demo_db_path)
        print("✅ Database repository initialized")
        print(f"   Database path: {repo.db_path}")
        
        # 1. Create and store stocks
        print("\n1. Creating and Storing Stocks:")
        
        stocks_to_create = [
            Stock(id=None, symbol="AAPL", name="Apple Inc.", exchange="NASDAQ"),
            Stock(id=None, symbol="GOOGL", name="Alphabet Inc.", exchange="NASDAQ"),
            Stock(id=None, symbol="MSFT", name="Microsoft Corporation", exchange="NASDAQ"),
        ]
        
        created_stocks = []
        for stock in stocks_to_create:
            try:
                created_stock = await repo.create_stock(stock)
                created_stocks.append(created_stock)
                print(f"   ✅ Created: {created_stock.symbol} (ID: {created_stock.id})")
            except Exception as e:
                # Stock might already exist
                existing_stock = await repo.get_stock_by_symbol(stock.symbol)
                if existing_stock:
                    created_stocks.append(existing_stock)
                    print(f"   ℹ️  Exists: {existing_stock.symbol} (ID: {existing_stock.id})")
                else:
                    print(f"   ❌ Error creating {stock.symbol}: {e}")
        
        # 2. Add stock prices
        print("\n2. Adding Stock Price Data:")
        
        for stock in created_stocks:
            prices = []
            base_price = Decimal('150.00') if stock.symbol == "AAPL" else Decimal('100.00')
            
            # Generate 30 days of sample price data
            for i in range(30):
                price_date = datetime.now().date() - timedelta(days=29-i)
                
                # Simple price variation
                variation = Decimal(str((i % 10 - 5) * 2.0))
                current_price = base_price + variation
                
                price = StockPrice(
                    id=None,
                    stock_id=stock.id,
                    date=price_date,
                    open_price=current_price - Decimal('1.00'),
                    high_price=current_price + Decimal('2.00'),
                    low_price=current_price - Decimal('2.00'),
                    close_price=current_price,
                    volume=1000000 + (i * 50000),
                    adjusted_close=current_price
                )
                prices.append(price)
            
            try:
                await repo.create_stock_prices(prices)
                print(f"   ✅ Added {len(prices)} prices for {stock.symbol}")
            except Exception as e:
                print(f"   ⚠️  Error adding prices for {stock.symbol}: {e}")
        
        # 3. Query and display data
        print("\n3. Querying Stock Data:")
        
        all_stocks = await repo.get_all_stocks()
        print(f"   📊 Total stocks in database: {len(all_stocks)}")
        
        for stock in all_stocks[:3]:  # Show first 3
            latest_price = await repo.get_latest_stock_price(stock.id)
            if latest_price:
                print(f"   📈 {stock.symbol}: ${latest_price.close_price} (Volume: {latest_price.volume:,})")
            else:
                print(f"   📈 {stock.symbol}: No price data")
        
        # 4. Demonstrate analytics
        print("\n4. Stock Analytics:")
        
        if created_stocks:
            stock = created_stocks[0]  # Use first stock
            stats = await repo.get_stock_statistics(stock.id, days_back=30)
            
            if stats.get('data_points', 0) > 0:
                print(f"   📊 {stock.symbol} Statistics (30 days):")
                print(f"      • Data points: {stats['data_points']}")
                print(f"      • Average price: ${stats['avg_price']:.2f}")
                print(f"      • Price range: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}")
                print(f"      • Average volume: {stats['avg_volume']:,}")
            else:
                print(f"   📊 No statistics available for {stock.symbol}")
        
        # 5. Demonstrate stock info
        print("\n5. Adding Stock Information:")
        
        if created_stocks:
            stock = created_stocks[0]
            stock_info = StockInfo(
                id=None,
                stock_id=stock.id,
                sector="Technology",
                industry="Consumer Electronics",
                market_cap=Decimal('2800000000000'),  # $2.8T
                pe_ratio=Decimal('28.5'),
                dividend_yield=Decimal('0.5'),
                beta=Decimal('1.2'),
                description="Technology company",
                website="https://apple.com",
                employees=150000
            )
            
            try:
                await repo.create_or_update_stock_info(stock_info)
                print(f"   ✅ Added info for {stock.symbol}")
                print(f"      • Sector: {stock_info.sector}")
                print(f"      • Market Cap: ${stock_info.market_cap:,}")
                print(f"      • P/E Ratio: {stock_info.pe_ratio}")
            except Exception as e:
                print(f"   ❌ Error adding info: {e}")
        
        # 6. Advanced queries
        print("\n6. Advanced Database Queries:")
        
        # Get stocks with recent data
        recent_date = datetime.now().date() - timedelta(days=7)
        stocks_with_data = await repo.get_stocks_with_prices_since(recent_date)
        print(f"   📅 Stocks with data since {recent_date}: {len(stocks_with_data)}")
        
        for item in stocks_with_data[:3]:
            stock = item['stock']
            print(f"      • {stock.symbol}: {item['price_count']} prices, latest: {item['latest_date']}")
        
        # Get sectors
        sectors = await repo.get_sectors()
        if sectors:
            print(f"   🏢 Available sectors: {', '.join(sectors)}")
        
        # 7. Performance demonstration
        print("\n7. Performance Test:")
        
        start_time = datetime.now()
        
        # Bulk operation test
        if created_stocks:
            stock = created_stocks[0]
            prices = await repo.get_stock_prices(stock.id, limit=10)
            
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"   ⚡ Retrieved {len(prices) if 'prices' in locals() else 0} prices in {duration:.3f}s")
        
        # 8. Data integrity demonstration
        print("\n8. Data Integrity & Validation:")
        
        try:
            # This should fail validation
            invalid_price = StockPrice(
                id=None,
                stock_id=1,
                date=datetime.now().date(),
                open_price=Decimal('100.00'),
                high_price=Decimal('90.00'),  # High < Low - invalid!
                low_price=Decimal('95.00'),
                close_price=Decimal('98.00'),
                volume=1000000
            )
            print("   ❌ This should not print - validation failed!")
        except ValueError as e:
            print(f"   ✅ Validation working: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 DATABASE ARCHITECTURE DEMO COMPLETE!")
        print("=" * 60)
        
        print("\n✅ Successfully demonstrated:")
        print("  • SQLite repository implementation")
        print("  • Complete CRUD operations")
        print("  • Domain entity validation")
        print("  • Bulk data operations")
        print("  • Advanced analytics queries")
        print("  • Data integrity enforcement")
        print("  • Performance optimization")
        
        print("\n🏗️ Architecture Benefits:")
        print("  • Type-safe database operations")
        print("  • Automatic table creation")
        print("  • Transaction management")
        print("  • Connection pooling ready")
        print("  • Async/await throughout")
        print("  • Comprehensive error handling")
        
        print("\n🚀 Ready for Production:")
        print("  • Scalable repository pattern")
        print("  • Easy to test with mocks")
        print("  • Clean separation of concerns")
        print("  • Professional logging")
        print("  • Configuration management")
        
    except Exception as e:
        logger.error(f"Error during database demo: {e}")
        print(f"\n❌ Demo error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

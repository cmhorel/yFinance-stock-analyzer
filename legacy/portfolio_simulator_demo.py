"""
Portfolio Simulator Demo - Using Clean Architecture

This demo shows how to use the new portfolio simulator service
with the clean architecture components for automated trading.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import clean architecture components
from shared.config import get_settings, setup_logging
from shared.logging import get_logger
from infrastructure.database import SqliteStockRepository
from infrastructure.database.sqlite_portfolio_repository import SqlitePortfolioRepository
from domain.services.portfolio_simulator_service import PortfolioSimulatorService
from domain.entities.stock import Stock, StockPrice, StockInfo
import yfinance as yf
import pandas as pd


class PortfolioSimulatorDemo:
    """Demo class for portfolio simulation."""
    
    def __init__(self):
        # Setup logging
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Initialize repositories
        self.stock_repo = SqliteStockRepository(db_path="data/portfolio_simulator.db")
        self.portfolio_repo = SqlitePortfolioRepository(db_path="data/portfolio_simulator.db")
        
        # Initialize simulator service
        self.simulator = PortfolioSimulatorService(
            stock_repo=self.stock_repo,
            portfolio_repo=self.portfolio_repo,
            initial_cash=Decimal('10000')
        )
        
        # Demo stock symbols
        self.demo_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
            'NVDA', 'META', 'NFLX', 'AMD', 'CRM'
        ]
    
    async def setup_demo_data(self):
        """Setup demo stock data for simulation."""
        self.logger.info("Setting up demo data...")
        
        for symbol in self.demo_symbols:
            try:
                # Check if stock already exists
                existing_stock = await self.stock_repo.get_stock_by_symbol(symbol)
                if existing_stock:
                    self.logger.info(f"Stock {symbol} already exists, skipping...")
                    continue
                
                # Create stock entity
                stock = Stock(
                    id=None,
                    symbol=symbol,
                    name=f"{symbol} Inc.",
                    exchange="NASDAQ"
                )
                
                created_stock = await self.stock_repo.create_stock(stock)
                self.logger.info(f"Created stock: {symbol}")
                
                # Fetch and store price data
                await self._fetch_stock_data(created_stock)
                
            except Exception as e:
                self.logger.error(f"Error setting up data for {symbol}: {e}")
    
    async def _fetch_stock_data(self, stock: Stock):
        """Fetch and store stock price data."""
        try:
            # Get data from yfinance
            ticker = yf.Ticker(stock.symbol)
            
            # Get 6 months of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                self.logger.warning(f"No price data found for {stock.symbol}")
                return
            
            # Convert to StockPrice entities
            prices = []
            for date, row in hist.iterrows():
                if pd.isna(row[['Open', 'High', 'Low', 'Close', 'Volume']]).any():
                    continue
                
                price = StockPrice(
                    id=None,
                    stock_id=stock.id,
                    date=date.date(),
                    open_price=Decimal(str(round(row['Open'], 2))),
                    high_price=Decimal(str(round(row['High'], 2))),
                    low_price=Decimal(str(round(row['Low'], 2))),
                    close_price=Decimal(str(round(row['Close'], 2))),
                    volume=int(row['Volume']),
                    adjusted_close=Decimal(str(round(row['Close'], 2)))
                )
                prices.append(price)
            
            if prices:
                await self.stock_repo.create_stock_prices(prices)
                self.logger.info(f"Stored {len(prices)} price records for {stock.symbol}")
            
            # Get and store stock info
            try:
                info = ticker.info
                if info and 'sector' in info:
                    stock_info = StockInfo(
                        id=None,
                        stock_id=stock.id,
                        sector=info.get('sector'),
                        industry=info.get('industry'),
                        market_cap=Decimal(str(info.get('marketCap', 0))) if info.get('marketCap') else None,
                        pe_ratio=Decimal(str(info.get('trailingPE', 0))) if info.get('trailingPE') else None,
                        dividend_yield=Decimal(str(info.get('dividendYield', 0))) if info.get('dividendYield') else None,
                        beta=Decimal(str(info.get('beta', 0))) if info.get('beta') else None,
                        description=info.get('longBusinessSummary'),
                        website=info.get('website'),
                        employees=info.get('fullTimeEmployees')
                    )
                    await self.stock_repo.create_or_update_stock_info(stock_info)
                    self.logger.info(f"Stored info for {stock.symbol}")
            except Exception as e:
                self.logger.warning(f"Could not fetch info for {stock.symbol}: {e}")
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {stock.symbol}: {e}")
    
    async def run_simulation(self, days: int = 5):
        """Run portfolio simulation for specified number of days."""
        self.logger.info(f"Starting {days}-day portfolio simulation...")
        
        # Initialize portfolio
        portfolio = await self.simulator.initialize_portfolio("Demo Portfolio")
        self.logger.info(f"Initialized portfolio with ${portfolio.cash_balance}")
        
        simulation_results = []
        
        for day in range(days):
            self.logger.info(f"\n--- Day {day + 1} of {days} ---")
            
            # Run trading simulation for this day
            day_results = await self.simulator.simulate_trading_day(self.demo_symbols)
            simulation_results.append(day_results)
            
            # Log results
            self.logger.info(f"Trades executed: {len(day_results['trades_executed'])}")
            self.logger.info(f"Portfolio value: ${day_results['portfolio_value_after']:.2f}")
            self.logger.info(f"Cash balance: ${day_results['cash_after']:.2f}")
            
            # Log individual trades
            for trade in day_results['trades_executed']:
                self.logger.info(f"  {trade['type']}: {trade['quantity']} shares of {trade['symbol']} at ${trade['price']:.2f}")
        
        # Final portfolio performance
        performance = await self.simulator.get_portfolio_performance()
        
        self.logger.info("\n=== SIMULATION COMPLETE ===")
        self.logger.info(f"Initial Value: ${performance['initial_value']:.2f}")
        self.logger.info(f"Final Value: ${performance['current_value']:.2f}")
        self.logger.info(f"Total Return: ${performance['total_return']:.2f}")
        self.logger.info(f"Return %: {performance['return_percentage']:.2f}%")
        self.logger.info(f"Holdings: {performance['holdings_count']} positions")
        self.logger.info(f"Total Transactions: {performance['total_transactions']}")
        
        return simulation_results, performance
    
    async def show_portfolio_status(self):
        """Show current portfolio status."""
        self.logger.info("\n=== PORTFOLIO STATUS ===")
        
        # Get portfolio
        portfolio = await self.simulator.get_current_portfolio()
        if not portfolio:
            self.logger.info("No portfolio found")
            return
        
        # Get performance
        performance = await self.simulator.get_portfolio_performance()
        
        self.logger.info(f"Portfolio Name: {portfolio.name}")
        self.logger.info(f"Cash Balance: ${performance['cash_balance']:.2f}")
        self.logger.info(f"Total Value: ${performance['current_value']:.2f}")
        self.logger.info(f"Total Return: ${performance['total_return']:.2f} ({performance['return_percentage']:.2f}%)")
        
        # Get holdings
        holdings = await self.portfolio_repo.get_current_holdings()
        if holdings:
            self.logger.info(f"\nCurrent Holdings ({len(holdings)} positions):")
            for holding in holdings:
                # Get current price
                latest_prices = await self.stock_repo.get_stock_prices(holding.stock_id, limit=1)
                current_price = latest_prices[0].close_price if latest_prices else Decimal('0')
                market_value = holding.quantity * current_price
                gain_loss = market_value - holding.total_cost
                gain_loss_pct = (gain_loss / holding.total_cost) * 100 if holding.total_cost > 0 else 0
                
                self.logger.info(f"  {holding.symbol}: {holding.quantity} shares @ ${holding.avg_cost_per_share:.2f}")
                self.logger.info(f"    Current: ${current_price:.2f} | Value: ${market_value:.2f} | P&L: ${gain_loss:.2f} ({gain_loss_pct:.1f}%)")
        
        # Get recent transactions
        transactions = await self.portfolio_repo.get_transactions(limit=5)
        if transactions:
            self.logger.info(f"\nRecent Transactions:")
            for tx in transactions:
                self.logger.info(f"  {tx.transaction_date.strftime('%Y-%m-%d')}: {tx.transaction_type.value} {tx.quantity} {tx.symbol} @ ${tx.price_per_share:.2f}")
    
    async def analyze_stocks(self):
        """Analyze stocks and show recommendations."""
        self.logger.info("\n=== STOCK ANALYSIS ===")
        
        analysis_results = await self.simulator.analyze_stocks_for_trading(self.demo_symbols)
        
        if not analysis_results:
            self.logger.info("No analysis results available")
            return
        
        # Get recommendations
        recommendations = self.simulator.analysis_service.get_recommendations(analysis_results)
        
        # Show buy recommendations
        buy_recs = recommendations.get("buy", [])
        if buy_recs:
            self.logger.info(f"\nBuy Recommendations ({len(buy_recs)}):")
            for i, rec in enumerate(buy_recs[:5], 1):
                self.logger.info(f"  {i}. {rec['symbol']}: Score {rec['score']}, Confidence {rec['confidence']:.2f}")
                self.logger.info(f"     Price: ${rec['price']:.2f}, Risk: {rec['risk_score']:.1f}")
                self.logger.info(f"     Reasons: {', '.join(rec['reasons'])}")
        
        # Show sell recommendations
        sell_recs = recommendations.get("sell", [])
        if sell_recs:
            self.logger.info(f"\nSell Recommendations ({len(sell_recs)}):")
            for i, rec in enumerate(sell_recs[:5], 1):
                self.logger.info(f"  {i}. {rec['symbol']}: Score {rec['score']}, Confidence {rec['confidence']:.2f}")
                self.logger.info(f"     Price: ${rec['price']:.2f}, Risk: {rec['risk_score']:.1f}")
                self.logger.info(f"     Reasons: {', '.join(rec['reasons'])}")


async def main():
    """Main demo function."""
    demo = PortfolioSimulatorDemo()
    
    print("🚀 Portfolio Simulator Demo - Clean Architecture")
    print("=" * 60)
    
    try:
        # Setup demo data
        await demo.setup_demo_data()
        
        # Show initial analysis
        await demo.analyze_stocks()
        
        # Run simulation
        results, performance = await demo.run_simulation(days=3)
        
        # Show final status
        await demo.show_portfolio_status()
        
        print("\n✅ Demo completed successfully!")
        print(f"📊 Database: data/portfolio_simulator.db")
        print(f"📈 Final Return: {performance['return_percentage']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

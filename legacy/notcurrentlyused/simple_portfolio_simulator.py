"""
Simple Portfolio Simulator - Using Clean Architecture

A simplified portfolio simulator that uses the new analysis tools
and clean architecture components for automated trading simulation.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any
import sqlite3

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import clean architecture components
from shared.config import get_settings, setup_logging
from shared.logging import get_logger
from infrastructure.database import SqliteStockRepository
from domain.services.stock_analysis_service import StockAnalysisService
from domain.entities.stock import Stock, StockPrice, StockInfo
import yfinance as yf
import pandas as pd


class SimplePortfolioSimulator:
    """Simple portfolio simulator using clean architecture."""
    
    def __init__(self, db_path: str = "data/simple_portfolio.db"):
        # Setup logging
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Initialize repositories and services
        self.stock_repo = SqliteStockRepository(db_path=db_path)
        self.analysis_service = StockAnalysisService()
        self.db_path = db_path
        
        # Trading parameters
        self.initial_cash = Decimal('10000')
        self.transaction_fee = Decimal('10')
        self.max_position_size = Decimal('0.1')  # 10% max per position
        
        # Initialize portfolio database
        self._init_portfolio_db()
        
        # Demo symbols
        self.demo_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
            'NVDA', 'META', 'NFLX', 'AMD', 'CRM'
        ]
    
    def _init_portfolio_db(self):
        """Initialize portfolio database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY,
                    cash_balance DECIMAL(15,2) NOT NULL,
                    total_value DECIMAL(15,2) NOT NULL,
                    created_date DATE NOT NULL,
                    updated_date DATE NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS holdings (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    avg_cost DECIMAL(10,4) NOT NULL,
                    total_cost DECIMAL(15,2) NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price DECIMAL(10,4) NOT NULL,
                    total_amount DECIMAL(15,2) NOT NULL,
                    fee DECIMAL(10,2) NOT NULL,
                    transaction_date DATE NOT NULL
                );
            """)
    
    def _get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM portfolio ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            
            if not row:
                # Initialize portfolio
                today = datetime.now().date()
                conn.execute(
                    "INSERT INTO portfolio (cash_balance, total_value, created_date, updated_date) VALUES (?, ?, ?, ?)",
                    (float(self.initial_cash), float(self.initial_cash), today, today)
                )
                conn.commit()
                return {
                    'cash_balance': self.initial_cash,
                    'total_value': self.initial_cash,
                    'created_date': today,
                    'updated_date': today
                }
            
            return {
                'cash_balance': Decimal(str(row[1])),
                'total_value': Decimal(str(row[2])),
                'created_date': row[3],
                'updated_date': row[4]
            }
    
    def _update_portfolio(self, cash_balance: Decimal, total_value: Decimal):
        """Update portfolio state."""
        with sqlite3.connect(self.db_path) as conn:
            today = datetime.now().date()
            conn.execute(
                "UPDATE portfolio SET cash_balance = ?, total_value = ?, updated_date = ? WHERE id = (SELECT MAX(id) FROM portfolio)",
                (float(cash_balance), float(total_value), today)
            )
            conn.commit()
    
    def _get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM holdings WHERE quantity > 0")
            holdings = []
            for row in cursor.fetchall():
                holdings.append({
                    'symbol': row[0],
                    'quantity': row[1],
                    'avg_cost': Decimal(str(row[2])),
                    'total_cost': Decimal(str(row[3]))
                })
            return holdings
    
    def _execute_buy(self, symbol: str, quantity: int, price: Decimal) -> bool:
        """Execute a buy transaction."""
        try:
            total_cost = (quantity * price) + self.transaction_fee
            portfolio = self._get_portfolio()
            
            if total_cost > portfolio['cash_balance']:
                self.logger.warning(f"Insufficient funds for {symbol}: need {total_cost}, have {portfolio['cash_balance']}")
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                # Record transaction
                conn.execute(
                    "INSERT INTO transactions (symbol, transaction_type, quantity, price, total_amount, fee, transaction_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (symbol, 'BUY', quantity, float(price), float(total_cost), float(self.transaction_fee), datetime.now().date())
                )
                
                # Update or create holding
                cursor = conn.execute("SELECT * FROM holdings WHERE symbol = ?", (symbol,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing holding
                    old_quantity = existing[1]
                    old_total_cost = Decimal(str(existing[3]))
                    new_quantity = old_quantity + quantity
                    new_total_cost = old_total_cost + (quantity * price)
                    new_avg_cost = new_total_cost / new_quantity
                    
                    conn.execute(
                        "UPDATE holdings SET quantity = ?, avg_cost = ?, total_cost = ? WHERE symbol = ?",
                        (new_quantity, float(new_avg_cost), float(new_total_cost), symbol)
                    )
                else:
                    # Create new holding
                    conn.execute(
                        "INSERT INTO holdings (symbol, quantity, avg_cost, total_cost) VALUES (?, ?, ?, ?)",
                        (symbol, quantity, float(price), float(quantity * price))
                    )
                
                # Update portfolio cash
                new_cash = portfolio['cash_balance'] - total_cost
                new_total_value = self._calculate_portfolio_value(new_cash)
                self._update_portfolio(new_cash, new_total_value)
                
                conn.commit()
                self.logger.info(f"BUY: {quantity} shares of {symbol} at ${price:.2f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error executing buy for {symbol}: {e}")
            return False
    
    def _execute_sell(self, symbol: str, quantity: int, price: Decimal) -> bool:
        """Execute a sell transaction."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check holding
                cursor = conn.execute("SELECT * FROM holdings WHERE symbol = ?", (symbol,))
                holding = cursor.fetchone()
                
                if not holding or holding[1] < quantity:
                    self.logger.warning(f"Insufficient shares for {symbol}: need {quantity}, have {holding[1] if holding else 0}")
                    return False
                
                gross_proceeds = quantity * price
                net_proceeds = gross_proceeds - self.transaction_fee
                
                # Record transaction
                conn.execute(
                    "INSERT INTO transactions (symbol, transaction_type, quantity, price, total_amount, fee, transaction_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (symbol, 'SELL', quantity, float(price), float(gross_proceeds), float(self.transaction_fee), datetime.now().date())
                )
                
                # Update holding
                new_quantity = holding[1] - quantity
                if new_quantity == 0:
                    conn.execute("DELETE FROM holdings WHERE symbol = ?", (symbol,))
                else:
                    avg_cost = Decimal(str(holding[2]))
                    new_total_cost = new_quantity * avg_cost
                    conn.execute(
                        "UPDATE holdings SET quantity = ?, total_cost = ? WHERE symbol = ?",
                        (new_quantity, float(new_total_cost), symbol)
                    )
                
                # Update portfolio cash
                portfolio = self._get_portfolio()
                new_cash = portfolio['cash_balance'] + net_proceeds
                new_total_value = self._calculate_portfolio_value(new_cash)
                self._update_portfolio(new_cash, new_total_value)
                
                conn.commit()
                self.logger.info(f"SELL: {quantity} shares of {symbol} at ${price:.2f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error executing sell for {symbol}: {e}")
            return False
    
    def _calculate_portfolio_value(self, cash_balance: Decimal = None) -> Decimal:
        """Calculate total portfolio value."""
        if cash_balance is None:
            portfolio = self._get_portfolio()
            cash_balance = portfolio['cash_balance']
        
        total_value = cash_balance
        holdings = self._get_holdings()
        
        for holding in holdings:
            # For simplicity, use the average cost as current value
            # In a real system, you'd get current market prices
            total_value += holding['total_cost']
        
        return total_value
    
    async def setup_demo_data(self):
        """Setup demo stock data."""
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
    
    async def analyze_stocks(self) -> List[Dict[str, Any]]:
        """Analyze stocks and return recommendations."""
        self.logger.info("Analyzing stocks for trading opportunities...")
        
        analysis_results = []
        
        for symbol in self.demo_symbols:
            try:
                # Get stock data
                stock = await self.stock_repo.get_stock_by_symbol(symbol)
                if not stock:
                    continue
                
                # Get recent price data
                prices = await self.stock_repo.get_stock_prices(stock.id, limit=200)
                if len(prices) < 50:
                    continue
                
                # Perform analysis
                analysis = self.analysis_service.analyze_stock(stock, prices)
                if analysis:
                    analysis_results.append(analysis)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Get recommendations
        if analysis_results:
            recommendations = self.analysis_service.get_recommendations(analysis_results)
            return recommendations
        
        return {"buy": [], "sell": []}
    
    async def simulate_trading_day(self) -> Dict[str, Any]:
        """Simulate one day of trading."""
        self.logger.info("Starting trading simulation...")
        
        portfolio_before = self._get_portfolio()
        trades_executed = []
        
        # Get recommendations
        recommendations = await self.analyze_stocks()
        
        # Execute buy orders (top 3)
        buy_recs = recommendations.get("buy", [])[:3]
        for rec in buy_recs:
            if rec["confidence"] > 0.6 and rec["risk_score"] < 5.0:
                # Calculate quantity (simple approach)
                price = Decimal(str(rec["price"]))
                max_investment = portfolio_before['cash_balance'] * self.max_position_size
                quantity = int(max_investment / price)
                
                if quantity > 0 and (quantity * price) > 100:  # Minimum $100 trade
                    if self._execute_buy(rec["symbol"], quantity, price):
                        trades_executed.append({
                            "type": "BUY",
                            "symbol": rec["symbol"],
                            "quantity": quantity,
                            "price": float(price),
                            "confidence": rec["confidence"]
                        })
        
        # Execute sell orders for holdings with sell signals
        holdings = self._get_holdings()
        sell_recs = recommendations.get("sell", [])
        
        for rec in sell_recs:
            if rec["confidence"] > 0.7:
                # Check if we own this stock
                holding = next((h for h in holdings if h["symbol"] == rec["symbol"]), None)
                if holding:
                    price = Decimal(str(rec["price"]))
                    quantity = holding["quantity"]
                    
                    if self._execute_sell(rec["symbol"], quantity, price):
                        trades_executed.append({
                            "type": "SELL",
                            "symbol": rec["symbol"],
                            "quantity": quantity,
                            "price": float(price),
                            "confidence": rec["confidence"]
                        })
        
        portfolio_after = self._get_portfolio()
        
        results = {
            "date": datetime.now().date(),
            "trades_executed": trades_executed,
            "portfolio_value_before": float(portfolio_before['cash_balance']),
            "portfolio_value_after": float(portfolio_after['cash_balance']),
            "total_trades": len(trades_executed)
        }
        
        self.logger.info(f"Trading day complete: {len(trades_executed)} trades executed")
        return results
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio performance summary."""
        portfolio = self._get_portfolio()
        holdings = self._get_holdings()
        
        # Get transaction count
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM transactions")
            transaction_count = cursor.fetchone()[0]
        
        total_value = self._calculate_portfolio_value()
        total_return = total_value - self.initial_cash
        return_percentage = (total_return / self.initial_cash) * 100
        
        return {
            "initial_value": float(self.initial_cash),
            "current_value": float(total_value),
            "cash_balance": float(portfolio['cash_balance']),
            "total_return": float(total_return),
            "return_percentage": float(return_percentage),
            "holdings_count": len(holdings),
            "total_transactions": transaction_count,
            "created_date": portfolio['created_date'],
            "updated_date": portfolio['updated_date']
        }


async def main():
    """Main demo function."""
    simulator = SimplePortfolioSimulator()
    
    print("🚀 Simple Portfolio Simulator - Clean Architecture")
    print("=" * 60)
    
    try:
        # Setup demo data
        await simulator.setup_demo_data()
        
        # Show initial analysis
        recommendations = await simulator.analyze_stocks()
        print(f"\n📊 Analysis Results:")
        print(f"   Buy recommendations: {len(recommendations.get('buy', []))}")
        print(f"   Sell recommendations: {len(recommendations.get('sell', []))}")
        
        # Run simulation for 3 days
        for day in range(3):
            print(f"\n--- Day {day + 1} ---")
            results = await simulator.simulate_trading_day()
            print(f"Trades executed: {results['total_trades']}")
            for trade in results['trades_executed']:
                print(f"  {trade['type']}: {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
        
        # Show final performance
        summary = simulator.get_portfolio_summary()
        print(f"\n=== FINAL RESULTS ===")
        print(f"Initial Value: ${summary['initial_value']:.2f}")
        print(f"Final Value: ${summary['current_value']:.2f}")
        print(f"Total Return: ${summary['total_return']:.2f} ({summary['return_percentage']:.2f}%)")
        print(f"Holdings: {summary['holdings_count']} positions")
        print(f"Total Transactions: {summary['total_transactions']}")
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"📊 Database: data/simple_portfolio.db")
        
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

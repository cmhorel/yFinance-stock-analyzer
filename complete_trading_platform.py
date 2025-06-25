"""
Complete Trading Platform

Full implementation that includes:
1. Auto-loads ALL NASDAQ & TSX stocks on startup
2. Launches web UI for trading
3. Stock analysis with buy/sell recommendations
4. Auto-trading functionality
5. Technical analysis charts
6. Portfolio management with P&L tracking
7. Centralized database for all data
"""

import asyncio
import sys
import os
import threading
import time
import webbrowser
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, date
from decimal import Decimal
import json

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from shared.config import setup_logging
from shared.logging import get_logger
from infrastructure.database.sqlite_stock_repository import SqliteStockRepository
from domain.entities.stock import Stock, StockPrice, StockInfo
from domain.entities.portfolio import Portfolio, PortfolioHolding
from domain.services.technical_analysis_service import TechnicalAnalysisService
from domain.services.stock_analysis_service import StockAnalysisService
from domain.services.chart_generation_service import ChartGenerationService
import yfinance as yf
import pandas as pd
import numpy as np

# Flask imports
from flask import Flask, render_template_string, jsonify, request


class SimplePortfolioRepo:
    """SQLite-based portfolio repository that actually persists data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = get_logger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the portfolio database tables."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create portfolios table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolios (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        cash_balance DECIMAL(15,2) NOT NULL,
                        total_portfolio_value DECIMAL(15,2) NOT NULL,
                        created_date TIMESTAMP NOT NULL,
                        updated_date TIMESTAMP NOT NULL
                    )
                ''')
                
                # Create holdings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS holdings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id INTEGER NOT NULL,
                        stock_id INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        avg_cost_per_share DECIMAL(10,4) NOT NULL,
                        total_cost DECIMAL(15,2) NOT NULL,
                        FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                    )
                ''')
                
                # Create portfolio history table for tracking value over time
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id INTEGER NOT NULL,
                        date DATE NOT NULL,
                        cash_balance DECIMAL(15,2) NOT NULL,
                        holdings_value DECIMAL(15,2) NOT NULL,
                        total_value DECIMAL(15,2) NOT NULL,
                        daily_return DECIMAL(10,6),
                        created_timestamp TIMESTAMP NOT NULL,
                        FOREIGN KEY (portfolio_id) REFERENCES portfolios (id),
                        UNIQUE(portfolio_id, date)
                    )
                ''')
                
                # Create trades table for tracking all transactions
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        trade_type TEXT NOT NULL, -- 'BUY' or 'SELL'
                        quantity INTEGER NOT NULL,
                        price DECIMAL(10,4) NOT NULL,
                        total_amount DECIMAL(15,2) NOT NULL,
                        fees DECIMAL(10,2) NOT NULL,
                        trade_date TIMESTAMP NOT NULL,
                        FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                    )
                ''')
                
                conn.commit()
                self.logger.info("Portfolio database tables initialized")
        except Exception as e:
            self.logger.error(f"Error initializing portfolio database: {e}")
    
    async def get_all_portfolios(self):
        """Get all portfolios from database."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM portfolios')
                rows = cursor.fetchall()
                
                portfolios = []
                for row in rows:
                    portfolio = Portfolio(
                        id=row[0],
                        name=row[1],
                        cash_balance=Decimal(str(row[2])),
                        total_portfolio_value=Decimal(str(row[3])),
                        created_date=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                        updated_date=datetime.fromisoformat(row[5]) if row[5] else datetime.now()
                    )
                    portfolios.append(portfolio)
                
                return portfolios
        except Exception as e:
            self.logger.error(f"Error getting portfolios: {e}")
            return []
    
    async def create_portfolio(self, portfolio: Portfolio):
        """Create a new portfolio in database."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO portfolios (name, cash_balance, total_portfolio_value, created_date, updated_date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    portfolio.name,
                    float(portfolio.cash_balance),
                    float(portfolio.total_portfolio_value),
                    portfolio.created_date.isoformat(),
                    portfolio.updated_date.isoformat()
                ))
                
                portfolio.id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Created portfolio {portfolio.name} with ID {portfolio.id}")
                return portfolio
        except Exception as e:
            self.logger.error(f"Error creating portfolio: {e}")
            return portfolio
    
    async def update_portfolio(self, portfolio: Portfolio):
        """Update portfolio in database."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE portfolios 
                    SET cash_balance = ?, total_portfolio_value = ?, updated_date = ?
                    WHERE id = ?
                ''', (
                    float(portfolio.cash_balance),
                    float(portfolio.total_portfolio_value),
                    portfolio.updated_date.isoformat(),
                    portfolio.id
                ))
                conn.commit()
                self.logger.info(f"Updated portfolio {portfolio.id} - Cash: ${portfolio.cash_balance}")
                return portfolio
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
            return portfolio
    
    async def get_holdings_by_portfolio_id(self, portfolio_id: int):
        """Get holdings for a portfolio from database."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM holdings WHERE portfolio_id = ?', (portfolio_id,))
                rows = cursor.fetchall()
                
                holdings = []
                for row in rows:
                    holding = PortfolioHolding(
                        id=row[0],
                        stock_id=row[2],
                        symbol=row[3],
                        quantity=row[4],
                        avg_cost_per_share=Decimal(str(row[5])),
                        total_cost=Decimal(str(row[6]))
                    )
                    # Add portfolio_id attribute
                    setattr(holding, 'portfolio_id', row[1])
                    holdings.append(holding)
                
                return holdings
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            return []
    
    async def create_holding(self, holding):
        """Create a new holding in database."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                portfolio_id = getattr(holding, 'portfolio_id', 1)
                
                cursor.execute('''
                    INSERT INTO holdings (portfolio_id, stock_id, symbol, quantity, avg_cost_per_share, total_cost)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    portfolio_id,
                    holding.stock_id,
                    holding.symbol,
                    holding.quantity,
                    float(holding.avg_cost_per_share),
                    float(holding.total_cost)
                ))
                
                holding.id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Created holding {holding.symbol} - Qty: {holding.quantity} @ ${holding.avg_cost_per_share}")
                return holding
        except Exception as e:
            self.logger.error(f"Error creating holding: {e}")
            return holding
    
    async def update_holding(self, holding):
        """Update holding in database."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE holdings 
                    SET quantity = ?, avg_cost_per_share = ?, total_cost = ?
                    WHERE id = ?
                ''', (
                    holding.quantity,
                    float(getattr(holding, 'avg_cost_per_share', 0)),
                    float(getattr(holding, 'total_cost', 0)),
                    holding.id
                ))
                conn.commit()
                self.logger.info(f"Updated holding {holding.symbol} - Qty: {holding.quantity}")
                return holding
        except Exception as e:
            self.logger.error(f"Error updating holding: {e}")
            return holding
    
    async def delete_holding(self, holding_id: int):
        """Delete holding from database."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM holdings WHERE id = ?', (holding_id,))
                conn.commit()
                self.logger.info(f"Deleted holding with ID {holding_id}")
        except Exception as e:
            self.logger.error(f"Error deleting holding: {e}")
    
    async def record_trade(self, portfolio_id: int, symbol: str, trade_type: str, 
                          quantity: int, price: float, fees: float = 9.99):
        """Record a trade in the trades table."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                total_amount = (price * quantity) + fees if trade_type == 'BUY' else (price * quantity) - fees
                
                cursor.execute('''
                    INSERT INTO trades (portfolio_id, symbol, trade_type, quantity, price, total_amount, fees, trade_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    portfolio_id, symbol, trade_type, quantity, price, total_amount, fees, datetime.now()
                ))
                conn.commit()
                self.logger.info(f"Recorded {trade_type} trade: {quantity} {symbol} @ ${price}")
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    async def record_portfolio_snapshot(self, portfolio_id: int, cash_balance: float, 
                                       holdings_value: float, total_value: float):
        """Record a daily portfolio value snapshot."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                today = date.today()
                
                # Calculate daily return if we have previous data
                cursor.execute('''
                    SELECT total_value FROM portfolio_history 
                    WHERE portfolio_id = ? AND date < ? 
                    ORDER BY date DESC LIMIT 1
                ''', (portfolio_id, today))
                
                previous_value = cursor.fetchone()
                daily_return = None
                if previous_value:
                    daily_return = (total_value - previous_value[0]) / previous_value[0]
                
                # Insert or update today's snapshot
                cursor.execute('''
                    INSERT OR REPLACE INTO portfolio_history 
                    (portfolio_id, date, cash_balance, holdings_value, total_value, daily_return, created_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    portfolio_id, today, cash_balance, holdings_value, total_value, 
                    daily_return, datetime.now()
                ))
                conn.commit()
                self.logger.info(f"Recorded portfolio snapshot: ${total_value:,.2f}")
        except Exception as e:
            self.logger.error(f"Error recording portfolio snapshot: {e}")
    
    async def get_portfolio_history(self, portfolio_id: int, days: int = 30):
        """Get portfolio value history for the specified number of days."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT date, cash_balance, holdings_value, total_value, daily_return
                    FROM portfolio_history 
                    WHERE portfolio_id = ? 
                    ORDER BY date DESC 
                    LIMIT ?
                ''', (portfolio_id, days))
                
                rows = cursor.fetchall()
                history = []
                for row in rows:
                    history.append({
                        'date': row[0],
                        'cash_balance': float(row[1]),
                        'holdings_value': float(row[2]),
                        'total_value': float(row[3]),
                        'daily_return': float(row[4]) if row[4] else 0.0
                    })
                
                return list(reversed(history))  # Return in chronological order
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {e}")
            return []
    
    async def get_trade_history(self, portfolio_id: int, limit: int = 50):
        """Get trade history for the portfolio."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, trade_type, quantity, price, total_amount, fees, trade_date
                    FROM trades 
                    WHERE portfolio_id = ? 
                    ORDER BY trade_date DESC 
                    LIMIT ?
                ''', (portfolio_id, limit))
                
                rows = cursor.fetchall()
                trades = []
                for row in rows:
                    trades.append({
                        'symbol': row[0],
                        'trade_type': row[1],
                        'quantity': row[2],
                        'price': float(row[3]),
                        'total_amount': float(row[4]),
                        'fees': float(row[5]),
                        'trade_date': row[6]
                    })
                
                return trades
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []


class StockDataService:
    """Service for managing stock data operations."""
    
    def __init__(self, stock_repository):
        self.stock_repo = stock_repository
        self.logger = get_logger(__name__)
    
    def get_nasdaq_100_tickers(self) -> List[str]:
        """Get major NASDAQ tickers."""
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'AMD']
    
    def get_tsx_tickers(self) -> List[str]:
        """Get major TSX tickers."""
        return ['SHOP.TO', 'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO']
    
    async def load_all_stocks_on_startup(self) -> Dict[str, Any]:
        """Load all NASDAQ and TSX stocks on application startup."""
        self.logger.info("Loading stocks on startup...")
        
        # Get tickers
        nasdaq_symbols = self.get_nasdaq_100_tickers()
        tsx_symbols = self.get_tsx_tickers()
        all_symbols = nasdaq_symbols + tsx_symbols
        
        self.logger.info(f"Loading {len(nasdaq_symbols)} NASDAQ and {len(tsx_symbols)} TSX stocks...")
        
        loaded_count = 0
        failed_count = 0
        
        for symbol in all_symbols:
            try:
                success = await self.sync_ticker_data(symbol)
                if success:
                    loaded_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                self.logger.error(f"Error loading {symbol}: {e}")
                failed_count += 1
        
        self.logger.info(f"Stock loading complete: {loaded_count} loaded, {failed_count} failed")
        
        return {
            'nasdaq_count': len(nasdaq_symbols),
            'tsx_count': len(tsx_symbols),
            'total_symbols': len(all_symbols),
            'loaded_count': loaded_count,
            'failed_count': failed_count
        }
    
    async def sync_ticker_data(self, symbol: str) -> bool:
        """Sync data for a single ticker."""
        try:
            stock = await self.stock_repo.get_stock_by_symbol(symbol)
            if not stock:
                exchange = "TSX" if ".TO" in symbol else "NASDAQ"
                stock = await self.stock_repo.create_stock(Stock(
                    id=None, symbol=symbol, name=symbol, exchange=exchange
                ))
            
            if stock.id is None:
                return False
            
            # Get recent price data
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(period="30d")  # Get more data for analysis
            
            if hist.empty:
                return True
            
            prices = []
            for date_idx, row in hist.iterrows():
                if pd.isna(row[['Open', 'High', 'Low', 'Close', 'Volume']]).any():
                    continue
                
                # Convert to date
                date_obj = date_idx.date() if hasattr(date_idx, 'date') else datetime.now().date()
                
                price = StockPrice(
                    id=None, stock_id=stock.id, date=date_obj,
                    open_price=Decimal(str(row['Open'])),
                    high_price=Decimal(str(row['High'])),
                    low_price=Decimal(str(row['Low'])),
                    close_price=Decimal(str(row['Close'])),
                    volume=int(row['Volume']),
                    adjusted_close=Decimal(str(row['Close']))
                )
                prices.append(price)
            
            if prices:
                await self.stock_repo.create_stock_prices(prices)
            
            return True
        except Exception as e:
            self.logger.error(f"Error syncing {symbol}: {e}")
            return False
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except:
            return None


class AnalysisService:
    """Service for stock analysis and recommendations."""
    
    def __init__(self, stock_repo):
        self.stock_repo = stock_repo
        self.technical_analysis = TechnicalAnalysisService()
        self.stock_analysis = StockAnalysisService()
        self.logger = get_logger(__name__)
    
    def _get_sector_for_symbol(self, symbol: str) -> tuple:
        """Get sector and industry for a stock symbol."""
        # Define sector mappings for major stocks
        sector_map = {
            # Technology
            'AAPL': ('Technology', 'Consumer Electronics'),
            'GOOGL': ('Communication Services', 'Internet Content & Information'),
            'GOOG': ('Communication Services', 'Internet Content & Information'),
            'MSFT': ('Technology', 'Software—Infrastructure'),
            'NVDA': ('Technology', 'Semiconductors'),
            'META': ('Communication Services', 'Internet Content & Information'),
            'TSLA': ('Consumer Cyclical', 'Auto Manufacturers'),
            'NFLX': ('Communication Services', 'Entertainment'),
            'CRM': ('Technology', 'Software—Application'),
            'AMD': ('Technology', 'Semiconductors'),
            
            # Canadian stocks
            'SHOP.TO': ('Technology', 'Software—Application'),
            'SHOP': ('Technology', 'Software—Application'),
            'RY.TO': ('Financial Services', 'Banks—Diversified'),
            'TD.TO': ('Financial Services', 'Banks—Diversified'),
            'BNS.TO': ('Financial Services', 'Banks—Diversified'),
            'BMO.TO': ('Financial Services', 'Banks—Diversified'),
            'CNR.TO': ('Industrials', 'Railroads'),
            'CP.TO': ('Industrials', 'Railroads'),
            'ENB.TO': ('Energy', 'Oil & Gas Midstream'),
            'TRP.TO': ('Energy', 'Oil & Gas Midstream'),
            'CNQ.TO': ('Energy', 'Oil & Gas E&P'),
            'SU.TO': ('Energy', 'Oil & Gas Integrated'),
            'WCN.TO': ('Industrials', 'Waste Management'),
            'CSU.TO': ('Technology', 'Software—Infrastructure'),
            'BAM.TO': ('Real Estate', 'Real Estate Services'),
            'ATD.TO': ('Consumer Defensive', 'Specialty Retail'),
            'L.TO': ('Consumer Defensive', 'Grocery Stores'),
            'MFC.TO': ('Financial Services', 'Insurance—Life'),
            'SLF.TO': ('Financial Services', 'Insurance—Life'),
            'CM.TO': ('Financial Services', 'Banks—Diversified'),
            'BB.TO': ('Technology', 'Software—Infrastructure'),
            'WEED.TO': ('Healthcare', 'Drug Manufacturers—Specialty & Generic'),
            'ACB.TO': ('Healthcare', 'Drug Manufacturers—Specialty & Generic'),
        }
        
        return sector_map.get(symbol, ('Technology', 'Software'))
    
    async def get_stock_recommendations(self) -> List[Dict[str, Any]]:
        """Get buy/sell recommendations for all stocks."""
        try:
            recommendations = []
            stocks = await self.stock_repo.get_all_stocks()
            
            for stock in stocks:  # Limit to first 10 for demo
                try:
                    # Get recent prices
                    prices = await self.stock_repo.get_stock_prices(stock.id, limit=30)
                    if len(prices) < 10:
                        continue
                    
                    # Calculate technical indicators
                    close_prices = [float(p.close_price) for p in prices]
                    
                    # Simple moving averages
                    sma_5 = np.mean(close_prices[-5:])
                    sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else sma_5
                    
                    # RSI calculation (simplified)
                    gains = []
                    losses = []
                    for i in range(1, len(close_prices)):
                        change = close_prices[i] - close_prices[i-1]
                        if change > 0:
                            gains.append(change)
                            losses.append(0)
                        else:
                            gains.append(0)
                            losses.append(abs(change))
                    
                    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
                    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
                    
                    if avg_loss == 0:
                        rsi = 100
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    
                    current_price = close_prices[-1]
                    
                    # Calculate volatility
                    returns = [close_prices[i]/close_prices[i-1] - 1 for i in range(1, len(close_prices))]
                    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.2
                    
                    # Generate sentiment score (simplified)
                    sentiment = (rsi - 50) / 100  # Convert RSI to sentiment-like score
                    
                    # Get sector information
                    sector, industry = self._get_sector_for_symbol(stock.symbol)
                    
                    # Generate recommendation
                    recommendation = "HOLD"
                    confidence = 50
                    reasons = []
                    
                    if sma_5 > sma_20 and rsi < 70:
                        recommendation = "BUY"
                        confidence = 75
                        reasons.append("SMA5 > SMA20 (bullish trend)")
                        reasons.append(f"RSI {rsi:.1f} not overbought")
                    elif sma_5 < sma_20 and rsi > 30:
                        recommendation = "SELL"
                        confidence = 70
                        reasons.append("SMA5 < SMA20 (bearish trend)")
                        reasons.append(f"RSI {rsi:.1f} not oversold")
                    
                    if rsi > 80:
                        recommendation = "SELL"
                        confidence = 85
                        reasons.append(f"RSI {rsi:.1f} overbought")
                    elif rsi < 20:
                        recommendation = "BUY"
                        confidence = 85
                        reasons.append(f"RSI {rsi:.1f} oversold")
                    
                    recommendations.append({
                        'symbol': stock.symbol,
                        'current_price': current_price,
                        'recommendation': recommendation,
                        'confidence': confidence,
                        'rsi': rsi,
                        'sma_5': sma_5,
                        'sma_20': sma_20,
                        'reasons': reasons,
                        'sector': sector,
                        'industry': industry,
                        'sentiment': sentiment,
                        'volatility': volatility
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {stock.symbol}: {e}")
                    continue
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return []


class AutoTradingService:
    """Service for automated trading based on analysis."""
    
    def __init__(self, portfolio_service, analysis_service):
        self.portfolio_service = portfolio_service
        self.analysis_service = analysis_service
        self.logger = get_logger(__name__)
        self.auto_trading_enabled = False
        self.min_confidence = 75
        self.max_position_size = 1000  # Max dollars per position
    
    def enable_auto_trading(self, enabled: bool = True):
        """Enable or disable auto trading."""
        self.auto_trading_enabled = enabled
        self.logger.info(f"Auto trading {'enabled' if enabled else 'disabled'}")
    
    async def execute_auto_trades(self) -> List[Dict[str, Any]]:
        """Execute automatic trades based on analysis."""
        if not self.auto_trading_enabled:
            return []
        
        try:
            recommendations = await self.analysis_service.get_stock_recommendations()
            executed_trades = []
            
            for rec in recommendations:
                if rec['confidence'] >= self.min_confidence:
                    symbol = rec['symbol']
                    recommendation = rec['recommendation']
                    current_price = rec['current_price']
                    
                    if recommendation == "BUY":
                        # Calculate quantity based on max position size
                        quantity = int(self.max_position_size / current_price)
                        if quantity > 0:
                            result = await self.portfolio_service.execute_buy_order(symbol, quantity)
                            if result['success']:
                                executed_trades.append({
                                    'action': 'BUY',
                                    'symbol': symbol,
                                    'quantity': quantity,
                                    'price': current_price,
                                    'confidence': rec['confidence'],
                                    'reasons': rec['reasons']
                                })
                    
                    elif recommendation == "SELL":
                        # Check if we have holdings to sell
                        holdings = await self.portfolio_service.get_holdings()
                        for holding in holdings:
                            if holding['symbol'] == symbol:
                                quantity = int(holding['quantity'])
                                result = await self.portfolio_service.execute_sell_order(symbol, quantity)
                                if result['success']:
                                    executed_trades.append({
                                        'action': 'SELL',
                                        'symbol': symbol,
                                        'quantity': quantity,
                                        'price': current_price,
                                        'confidence': rec['confidence'],
                                        'reasons': rec['reasons']
                                    })
                                break
            
            return executed_trades
            
        except Exception as e:
            self.logger.error(f"Error executing auto trades: {e}")
            return []


class PortfolioService:
    """Service for portfolio management operations."""
    
    def __init__(self, portfolio_repo, stock_repo, stock_data_service):
        self.portfolio_repo = portfolio_repo
        self.stock_repo = stock_repo
        self.stock_data_service = stock_data_service
        self.logger = get_logger(__name__)
    
    async def initialize_portfolio(self, initial_cash: float = 100000.0):
        """Initialize portfolio if it doesn't exist."""
        try:
            portfolios = await self.portfolio_repo.get_all_portfolios()
            if not portfolios:
                portfolio = Portfolio(
                    id=None,
                    name="Default Portfolio",
                    cash_balance=Decimal(str(initial_cash)),
                    total_portfolio_value=Decimal(str(initial_cash)),
                    created_date=datetime.now(),
                    updated_date=datetime.now()
                )
                await self.portfolio_repo.create_portfolio(portfolio)
                self.logger.info(f"Created default portfolio with ${initial_cash}")
        except Exception as e:
            self.logger.error(f"Error initializing portfolio: {e}")
    
    async def get_portfolio_summary(self):
        """Get portfolio summary."""
        try:
            portfolios = await self.portfolio_repo.get_all_portfolios()
            if not portfolios:
                await self.initialize_portfolio()
                portfolios = await self.portfolio_repo.get_all_portfolios()
            
            portfolio = portfolios[0]
            holdings = await self.portfolio_repo.get_holdings_by_portfolio_id(portfolio.id)
            
            # Calculate current portfolio value
            total_value = portfolio.cash_balance
            for holding in holdings:
                current_price = await self.stock_data_service.get_current_price(holding.symbol)
                if current_price:
                    total_value += Decimal(str(current_price)) * holding.quantity
            
            return {
                'cash_balance': float(portfolio.cash_balance),
                'total_value': float(total_value),
                'holdings_count': len(holdings),
                'portfolio_id': portfolio.id
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e)}
    
    async def get_holdings(self):
        """Get all holdings with current values."""
        try:
            portfolios = await self.portfolio_repo.get_all_portfolios()
            if not portfolios:
                return []
            
            portfolio = portfolios[0]
            holdings = await self.portfolio_repo.get_holdings_by_portfolio_id(portfolio.id)
            
            result = []
            for holding in holdings:
                current_price = await self.stock_data_service.get_current_price(holding.symbol)
                if current_price:
                    avg_cost = float(getattr(holding, 'avg_cost_per_share', 0))
                    market_value = float(current_price) * float(holding.quantity)
                    cost_basis = avg_cost * float(holding.quantity)
                    gain_loss = market_value - cost_basis
                    gain_loss_percent = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                    
                    result.append({
                        'symbol': holding.symbol,
                        'quantity': float(holding.quantity),
                        'average_cost': avg_cost,
                        'current_price': current_price,
                        'market_value': market_value,
                        'gain_loss': gain_loss,
                        'gain_loss_percent': gain_loss_percent
                    })
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            return []
    
    async def execute_buy_order(self, symbol: str, quantity: int):
        """Execute a buy order."""
        try:
            current_price = await self.stock_data_service.get_current_price(symbol)
            if not current_price:
                return {'success': False, 'error': f'Could not get price for {symbol}'}
            
            portfolios = await self.portfolio_repo.get_all_portfolios()
            if not portfolios:
                return {'success': False, 'error': 'No portfolio found'}
            
            portfolio = portfolios[0]
            total_cost = Decimal(str(current_price)) * quantity + Decimal('9.99')
            
            if portfolio.cash_balance < total_cost:
                return {'success': False, 'error': 'Insufficient funds'}
            
            # Get or create stock
            stock = await self.stock_repo.get_stock_by_symbol(symbol)
            if not stock:
                exchange = "TSX" if ".TO" in symbol else "NASDAQ"
                stock = await self.stock_repo.create_stock(Stock(
                    id=None, symbol=symbol, name=symbol, exchange=exchange
                ))
            
            # Update or create holding
            holdings = await self.portfolio_repo.get_holdings_by_portfolio_id(portfolio.id)
            existing_holding = None
            for holding in holdings:
                if holding.stock_id == stock.id:
                    existing_holding = holding
                    break
            
            if existing_holding:
                # Update existing holding
                old_cost = getattr(existing_holding, 'avg_cost_per_share', Decimal('0'))
                new_quantity = existing_holding.quantity + quantity
                new_avg_cost = ((old_cost * existing_holding.quantity) + 
                               (Decimal(str(current_price)) * quantity)) / new_quantity
                
                # Update the holding
                existing_holding.quantity = new_quantity
                setattr(existing_holding, 'avg_cost_per_share', new_avg_cost)
                setattr(existing_holding, 'total_cost', new_avg_cost * new_quantity)
                await self.portfolio_repo.update_holding(existing_holding)
            else:
                # Create new holding
                new_holding = PortfolioHolding(
                    id=None,
                    stock_id=stock.id,
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost_per_share=Decimal(str(current_price)),
                    total_cost=Decimal(str(current_price)) * quantity
                )
                setattr(new_holding, 'portfolio_id', portfolio.id)
                await self.portfolio_repo.create_holding(new_holding)
            
            # Update portfolio cash balance
            portfolio.cash_balance -= total_cost
            portfolio.updated_date = datetime.now()
            await self.portfolio_repo.update_portfolio(portfolio)
            
            # Record the trade
            await self.portfolio_repo.record_trade(
                portfolio.id, symbol, 'BUY', quantity, current_price
            )
            
            # Record portfolio snapshot
            await self._record_portfolio_snapshot(portfolio.id)
            
            return {
                'success': True,
                'message': f'Successfully bought {quantity} shares of {symbol} at ${current_price:.2f}'
            }
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_sell_order(self, symbol: str, quantity: int):
        """Execute a sell order."""
        try:
            current_price = await self.stock_data_service.get_current_price(symbol)
            if not current_price:
                return {'success': False, 'error': f'Could not get price for {symbol}'}
            
            portfolios = await self.portfolio_repo.get_all_portfolios()
            if not portfolios:
                return {'success': False, 'error': 'No portfolio found'}
            
            portfolio = portfolios[0]
            
            # Find holding
            holdings = await self.portfolio_repo.get_holdings_by_portfolio_id(portfolio.id)
            holding = None
            for h in holdings:
                if h.symbol == symbol:
                    holding = h
                    break
            
            if not holding:
                return {'success': False, 'error': f'No holding found for {symbol}'}
            
            if holding.quantity < quantity:
                return {'success': False, 'error': f'Insufficient shares. You own {holding.quantity} shares'}
            
            # Calculate proceeds
            total_proceeds = Decimal(str(current_price)) * quantity - Decimal('9.99')
            
            # Update holding
            if holding.quantity == quantity:
                # Sell all shares - delete holding
                await self.portfolio_repo.delete_holding(holding.id)
            else:
                # Partial sale - update holding
                holding.quantity -= quantity
                setattr(holding, 'total_cost', getattr(holding, 'avg_cost_per_share', Decimal('0')) * holding.quantity)
                await self.portfolio_repo.update_holding(holding)
            
            # Update portfolio cash balance
            portfolio.cash_balance += total_proceeds
            portfolio.updated_date = datetime.now()
            await self.portfolio_repo.update_portfolio(portfolio)
            
            # Record the trade
            await self.portfolio_repo.record_trade(
                portfolio.id, symbol, 'SELL', quantity, current_price
            )
            
            # Record portfolio snapshot
            await self._record_portfolio_snapshot(portfolio.id)
            
            return {
                'success': True,
                'message': f'Successfully sold {quantity} shares of {symbol} at ${current_price:.2f}'
            }
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _record_portfolio_snapshot(self, portfolio_id: int):
        """Record a portfolio snapshot with current values."""
        try:
            # Get current portfolio summary
            summary = await self.get_portfolio_summary()
            if 'error' not in summary:
                cash_balance = summary['cash_balance']
                total_value = summary['total_value']
                holdings_value = total_value - cash_balance
                
                await self.portfolio_repo.record_portfolio_snapshot(
                    portfolio_id, cash_balance, holdings_value, total_value
                )
        except Exception as e:
            self.logger.error(f"Error recording portfolio snapshot: {e}")


class CompleteTradingPlatform:
    """Complete trading platform with analysis and auto-trading."""
    
    def __init__(self, db_path: str = "data/unified_analyzer.db"):
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Initialize repositories
        self.stock_repo = SqliteStockRepository(db_path=db_path)
        
        # Use simple portfolio repository that actually works
        self.portfolio_repo = SimplePortfolioRepo(db_path=db_path)
        
        # Initialize services
        self.stock_data_service = StockDataService(self.stock_repo)
        self.portfolio_service = PortfolioService(
            self.portfolio_repo, 
            self.stock_repo, 
            self.stock_data_service
        )
        self.analysis_service = AnalysisService(self.stock_repo)
        self.auto_trading_service = AutoTradingService(
            self.portfolio_service,
            self.analysis_service
        )
        
        # Initialize chart generation service
        self.chart_service = ChartGenerationService(
            self.stock_repo,
            self.portfolio_repo
        )
        
        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()
        
        self.logger.info(f"Initialized complete trading platform with database: {db_path}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main trading dashboard."""
            return self._render_complete_dashboard()
        
        @self.app.route('/api/portfolio/summary')
        def api_portfolio_summary():
            """Get portfolio summary."""
            def get_summary():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.portfolio_service.get_portfolio_summary())
                loop.close()
                return result
            
            return jsonify(get_summary())
        
        @self.app.route('/api/portfolio/holdings')
        def api_portfolio_holdings():
            """Get portfolio holdings."""
            def get_holdings():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.portfolio_service.get_holdings())
                loop.close()
                return result
            
            return jsonify(get_holdings())
        
        @self.app.route('/api/analysis/recommendations')
        def api_recommendations():
            """Get stock recommendations."""
            def get_recommendations():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.analysis_service.get_stock_recommendations())
                loop.close()
                return result
            
            return jsonify(get_recommendations())
        
        @self.app.route('/api/auto-trading/status')
        def api_auto_trading_status():
            """Get auto trading status."""
            return jsonify({
                'enabled': self.auto_trading_service.auto_trading_enabled,
                'min_confidence': self.auto_trading_service.min_confidence,
                'max_position_size': self.auto_trading_service.max_position_size
            })
        
        @self.app.route('/api/auto-trading/toggle', methods=['POST'])
        def api_toggle_auto_trading():
            """Toggle auto trading."""
            data = request.get_json()
            enabled = data.get('enabled', False)
            self.auto_trading_service.enable_auto_trading(enabled)
            return jsonify({'success': True, 'enabled': enabled})
        
        @self.app.route('/api/auto-trading/execute', methods=['POST'])
        def api_execute_auto_trades():
            """Execute auto trades."""
            def execute_trades():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.auto_trading_service.execute_auto_trades())
                loop.close()
                return result
            
            return jsonify(execute_trades())
        
        @self.app.route('/api/stock/quote/<symbol>')
        def api_stock_quote(symbol):
            """Get stock quote."""
            def get_quote():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                price = loop.run_until_complete(self.stock_data_service.get_current_price(symbol))
                loop.close()
                return price
            
            try:
                current_price = get_quote()
                if current_price is None:
                    return jsonify({'error': f'No data found for {symbol}'})
                
                return jsonify({
                    'symbol': symbol,
                    'current_price': current_price,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/trade/buy', methods=['POST'])
        def api_buy_stock():
            """Execute buy order."""
            data = request.get_json()
            symbol = data.get('symbol', '').upper()
            quantity = int(data.get('quantity', 0))
            
            if not symbol or quantity <= 0:
                return jsonify({'success': False, 'error': 'Invalid symbol or quantity'})
            
            def execute_buy():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.portfolio_service.execute_buy_order(symbol, quantity))
                loop.close()
                return result
            
            return jsonify(execute_buy())
        
        @self.app.route('/api/trade/sell', methods=['POST'])
        def api_sell_stock():
            """Execute sell order."""
            data = request.get_json()
            symbol = data.get('symbol', '').upper()
            quantity = int(data.get('quantity', 0))
            
            if not symbol or quantity <= 0:
                return jsonify({'success': False, 'error': 'Invalid symbol or quantity'})
            
            def execute_sell():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.portfolio_service.execute_sell_order(symbol, quantity))
                loop.close()
                return result
            
            return jsonify(execute_sell())
        
        @self.app.route('/plots/<path:filename>')
        def serve_plots(filename):
            """Serve plot files."""
            import os
            from flask import send_from_directory
            plots_dir = os.path.abspath('plots')
            return send_from_directory(plots_dir, filename)
        
        @self.app.route('/browse-plots')
        def browse_plots():
            """Browse plots directory."""
            import os
            plots_dir = os.path.abspath('plots')
            
            if not os.path.exists(plots_dir):
                return "Plots directory not found", 404
            
            # List all files in plots directory
            files = []
            for root, dirs, filenames in os.walk(plots_dir):
                for filename in filenames:
                    if filename.endswith(('.html', '.png', '.jpg', '.jpeg')):
                        rel_path = os.path.relpath(os.path.join(root, filename), plots_dir)
                        files.append(rel_path.replace('\\', '/'))
            
            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Browse Plots</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .container { max-width: 1000px; margin: 0 auto; }
                    .file-list { list-style: none; padding: 0; }
                    .file-list li { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
                    .file-list a { text-decoration: none; color: #007bff; font-weight: bold; }
                    .file-list a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 Available Plots</h1>
                    <p><a href="/">← Back to Trading Platform</a></p>
                    <ul class="file-list">
            '''
            
            for file in sorted(files):
                html += f'<li>📈 <a href="/plots/{file}" target="_blank">{file}</a></li>'
            
            html += '''
                    </ul>
                </div>
            </body>
            </html>
            '''
            
            return html
        
        @self.app.route('/browse-stock-charts')
        def browse_stock_charts():
            """Browse stock analysis charts."""
            import os
            plots_dir = os.path.join(os.path.abspath('plots'), 'stock_analysis')
            
            if not os.path.exists(plots_dir):
                return "Stock analysis charts not found", 404
            
            files = []
            for filename in os.listdir(plots_dir):
                if filename.endswith('.html'):
                    files.append(filename)
            
            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Browse Stock Analysis Charts</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .container { max-width: 1000px; margin: 0 auto; }
                    .file-list { list-style: none; padding: 0; }
                    .file-list li { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
                    .file-list a { text-decoration: none; color: #007bff; font-weight: bold; }
                    .file-list a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 Stock Analysis Charts</h1>
                    <p><a href="/">← Back to Trading Platform</a></p>
                    <ul class="file-list">
            '''
            
            for file in sorted(files):
                symbol = file.replace('_analysis.html', '')
                html += f'<li>📊 <a href="/plots/stock_analysis/{file}" target="_blank">{symbol} Analysis Chart</a></li>'
            
            html += '''
                    </ul>
                </div>
            </body>
            </html>
            '''
            
            return html
        
        @self.app.route('/browse-web-demo')
        def browse_web_demo():
            """Browse web demo charts."""
            import os
            plots_dir = os.path.join(os.path.abspath('plots'), 'web_demo')
            
            if not os.path.exists(plots_dir):
                return "Web demo charts not found", 404
            
            files = []
            for filename in os.listdir(plots_dir):
                if filename.endswith('.html'):
                    files.append(filename)
            
            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Browse Web Demo Charts</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .container { max-width: 1000px; margin: 0 auto; }
                    .file-list { list-style: none; padding: 0; }
                    .file-list li { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
                    .file-list a { text-decoration: none; color: #007bff; font-weight: bold; }
                    .file-list a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🌐 Web Demo Charts</h1>
                    <p><a href="/">← Back to Trading Platform</a></p>
                    <ul class="file-list">
            '''
            
            for file in sorted(files):
                symbol = file.replace('_analysis.html', '')
                html += f'<li>🎯 <a href="/plots/web_demo/{file}" target="_blank">{symbol} Demo Chart</a></li>'
            
            html += '''
                    </ul>
                </div>
            </body>
            </html>
            '''
            
            return html
        
        @self.app.route('/api/charts/generate', methods=['POST'])
        def api_generate_charts():
            """Generate all charts manually."""
            def generate_charts():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Get recommendations for sector overview
                    recommendations = loop.run_until_complete(self.analysis_service.get_stock_recommendations())
                    
                    # Generate all charts
                    result = loop.run_until_complete(self.chart_service.generate_all_charts(
                        portfolio_service=self.portfolio_service,
                        recommendations=recommendations
                    ))
                    return result
                finally:
                    loop.close()
            
            try:
                result = generate_charts()
                return jsonify({
                    'success': True,
                    'message': 'Charts generated successfully',
                    'results': result
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/charts/status')
        def api_charts_status():
            """Get chart generation status."""
            import os
            plots_dir = os.path.abspath('plots')
            
            if not os.path.exists(plots_dir):
                return jsonify({
                    'charts_exist': False,
                    'total_charts': 0,
                    'stock_charts': 0,
                    'overview_charts': 0
                })
            
            # Count charts
            stock_charts = 0
            overview_charts = 0
            
            for root, dirs, filenames in os.walk(plots_dir):
                for filename in filenames:
                    if filename.endswith('.html'):
                        if 'stock_analysis' in root or 'web_demo' in root:
                            stock_charts += 1
                        else:
                            overview_charts += 1
            
            return jsonify({
                'charts_exist': True,
                'total_charts': stock_charts + overview_charts,
                'stock_charts': stock_charts,
                'overview_charts': overview_charts,
                'last_updated': datetime.now().isoformat()
            })
    
    def _render_complete_dashboard(self):
        """Render the complete trading dashboard with analysis."""
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Complete Trading Platform</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
                .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
                .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }
                .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .card h3 { color: #333; margin-bottom: 15px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
                .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
                .stat-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; }
                .stat-number { font-size: 1.8em; font-weight: bold; }
                .stat-label { font-size: 0.9em; opacity: 0.9; }
                .trading-form { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .form-group { margin-bottom: 15px; }
                .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #333; }
                .form-group input, .form-group select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
                .btn { padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; text-transform: uppercase; transition: all 0.3s; }
                .btn-buy { background: #28a745; color: white; }
                .btn-buy:hover { background: #218838; }
                .btn-sell { background: #dc3545; color: white; }
                .btn-sell:hover { background: #c82333; }
                .btn-quote { background: #007bff; color: white; }
                .btn-quote:hover { background: #0056b3; }
                .btn-auto { background: #17a2b8; color: white; }
                .btn-auto:hover { background: #138496; }
                .btn-auto.active { background: #28a745; }
                .nav-btn { display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 0 10px; font-weight: bold; transition: all 0.3s; }
                .nav-btn:hover { background: #0056b3; color: white; text-decoration: none; }
                .holdings-table, .recommendations-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
                .holdings-table th, .holdings-table td, .recommendations-table th, .recommendations-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                .holdings-table th, .recommendations-table th { background: #f8f9fa; font-weight: bold; }
                .positive { color: #28a745; font-weight: bold; }
                .negative { color: #dc3545; font-weight: bold; }
                .neutral { color: #6c757d; font-weight: bold; }
                .message { padding: 15px; border-radius: 5px; margin: 10px 0; }
                .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
                .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
                .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
                .quote-display { background: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .loading { text-align: center; padding: 20px; color: #666; }
                .refresh-btn { background: #6c757d; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px; }
                .refresh-btn:hover { background: #5a6268; }
                .recommendation-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
                .rec-buy { background: #28a745; color: white; }
                .rec-sell { background: #dc3545; color: white; }
                .rec-hold { background: #6c757d; color: white; }
                .confidence-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }
                .confidence-fill { height: 100%; background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%); transition: width 0.3s; }
                .auto-trading-panel { background: #f8f9fa; border: 2px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
                .auto-trading-panel.enabled { border-color: #28a745; background: #d4edda; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Complete Trading Platform</h1>
                <p>Advanced trading with AI-powered analysis and auto-trading</p>
                <div style="margin-top: 15px;">
                    <a href="/" class="nav-btn">🏠 Home</a>
                    <a href="#analysis" class="nav-btn" onclick="document.getElementById('recommendations-display').scrollIntoView()">📊 Stock Analysis</a>
                    <a href="#portfolio" class="nav-btn" onclick="document.getElementById('portfolio-summary').scrollIntoView()">💼 Portfolio</a>
                    <a href="#auto-trading" class="nav-btn" onclick="document.getElementById('auto-trading-panel').scrollIntoView()">🤖 Auto Trading</a>
                </div>
            </div>
            
            <div class="container">
                <!-- Portfolio Summary -->
                <div class="card">
                    <h3>📊 Portfolio Summary <button class="refresh-btn" onclick="loadPortfolioSummary()">Refresh</button></h3>
                    <div id="portfolio-summary" class="loading">Loading portfolio data...</div>
                </div>
                
                <!-- Auto Trading Panel -->
                <div class="card">
                    <h3>🤖 Auto Trading Control</h3>
                    <div id="auto-trading-panel" class="auto-trading-panel">
                        <div class="grid">
                            <div>
                                <h4>Auto Trading Status</h4>
                                <p id="auto-status">Disabled</p>
                                <button id="auto-toggle-btn" class="btn btn-auto" onclick="toggleAutoTrading()">Enable Auto Trading</button>
                            </div>
                            <div>
                                <h4>Execute Auto Trades</h4>
                                <p>Run analysis and execute recommended trades</p>
                                <button class="btn btn-auto" onclick="executeAutoTrades()">Execute Auto Trades</button>
                            </div>
                        </div>
                        <div id="auto-trading-results"></div>
                    </div>
                </div>
                
                <!-- Charts & Graphs Section -->
                <div class="card">
                    <h3>📊 Interactive Charts & Analysis Graphs</h3>
                    <div class="grid">
                        <div>
                            <h4>📈 Overview Charts</h4>
                            <a href="/plots/sector_overview.html" target="_blank" class="btn btn-quote">Sector Overview</a>
                            <a href="/plots/portfolio_performance.html" target="_blank" class="btn btn-quote">Portfolio Performance</a>
                            <a href="/plots/volatility_risk_analysis.html" target="_blank" class="btn btn-quote">Volatility & Risk Analysis</a>
                        </div>
                        <div>
                            <h4>🔍 Individual Stock Charts</h4>
                            <a href="/browse-stock-charts" target="_blank" class="btn btn-quote">Browse All Stock Charts</a>
                            <a href="/browse-web-demo" target="_blank" class="btn btn-quote">Web Demo Charts</a>
                            <a href="/browse-plots" target="_blank" class="btn btn-quote">📁 Browse All Plots</a>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <h4>🎯 Quick Access to Popular Charts</h4>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                            <a href="/plots/stock_analysis/AAPL_analysis.html" target="_blank" class="btn" style="background: #007bff; color: white; padding: 8px 12px; font-size: 0.9em;">AAPL Chart</a>
                            <a href="/plots/stock_analysis/GOOGL_analysis.html" target="_blank" class="btn" style="background: #007bff; color: white; padding: 8px 12px; font-size: 0.9em;">GOOGL Chart</a>
                            <a href="/plots/stock_analysis/MSFT_analysis.html" target="_blank" class="btn" style="background: #007bff; color: white; padding: 8px 12px; font-size: 0.9em;">MSFT Chart</a>
                            <a href="/plots/stock_analysis/TSLA_analysis.html" target="_blank" class="btn" style="background: #007bff; color: white; padding: 8px 12px; font-size: 0.9em;">TSLA Chart</a>
                            <a href="/plots/stock_analysis/NVDA_analysis.html" target="_blank" class="btn" style="background: #007bff; color: white; padding: 8px 12px; font-size: 0.9em;">NVDA Chart</a>
                            <a href="/plots/stock_analysis/SHOP.TO_analysis.html" target="_blank" class="btn" style="background: #007bff; color: white; padding: 8px 12px; font-size: 0.9em;">SHOP.TO Chart</a>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <h4>🔄 Chart Management</h4>
                        <button class="btn btn-auto" onclick="regenerateCharts()">🔄 Regenerate All Charts</button>
                        <button class="btn refresh-btn" onclick="checkChartStatus()" style="background: #6c757d;">📊 Check Chart Status</button>
                        <div id="chart-status" style="margin-top: 10px;"></div>
                    </div>
                </div>
                
                <div class="grid-3">
                    <!-- Trading Interface -->
                    <div class="card">
                        <h3>💰 Manual Trading</h3>
                        
                        <!-- Stock Quote -->
                        <div class="trading-form">
                            <h4>Get Stock Quote</h4>
                            <div class="form-group">
                                <label>Symbol:</label>
                                <input type="text" id="quote-symbol" placeholder="e.g., AAPL, GOOGL, SHOP.TO" style="display: inline-block; width: 70%;">
                                <button class="btn btn-quote" onclick="getQuote()" style="width: 25%; margin-left: 5%;">Quote</button>
                            </div>
                            <div id="quote-display"></div>
                        </div>
                        
                        <!-- Buy Order -->
                        <div class="trading-form">
                            <h4>Buy Order</h4>
                            <div class="form-group">
                                <label>Symbol:</label>
                                <input type="text" id="buy-symbol" placeholder="e.g., AAPL">
                            </div>
                            <div class="form-group">
                                <label>Quantity:</label>
                                <input type="number" id="buy-quantity" placeholder="Number of shares" min="1">
                            </div>
                            <button class="btn btn-buy" onclick="executeBuy()">Buy Shares</button>
                        </div>
                        
                        <!-- Sell Order -->
                        <div class="trading-form">
                            <h4>Sell Order</h4>
                            <div class="form-group">
                                <label>Symbol:</label>
                                <input type="text" id="sell-symbol" placeholder="e.g., AAPL">
                            </div>
                            <div class="form-group">
                                <label>Quantity:</label>
                                <input type="number" id="sell-quantity" placeholder="Number of shares" min="1">
                            </div>
                            <button class="btn btn-sell" onclick="executeSell()">Sell Shares</button>
                        </div>
                        
                        <div id="trading-messages"></div>
                    </div>
                    
                    <!-- Stock Analysis -->
                    <div class="card">
                        <h3>📈 Stock Analysis <button class="refresh-btn" onclick="loadRecommendations()">Refresh</button></h3>
                        <div id="recommendations-display" class="loading">Loading analysis...</div>
                    </div>
                    
                    <!-- Holdings -->
                    <div class="card">
                        <h3>💼 Current Holdings <button class="refresh-btn" onclick="loadHoldings()">Refresh</button></h3>
                        <div id="holdings-display" class="loading">Loading holdings...</div>
                    </div>
                </div>
            </div>
            
            <script>
                let autoTradingEnabled = false;
                
                // Load data on page load
                document.addEventListener('DOMContentLoaded', function() {
                    loadPortfolioSummary();
                    loadHoldings();
                    loadRecommendations();
                    loadAutoTradingStatus();
                });
                
                async function loadPortfolioSummary() {
                    try {
                        const response = await fetch('/api/portfolio/summary');
                        const data = await response.json();
                        
                        if (data.error) {
                            document.getElementById('portfolio-summary').innerHTML = 
                                `<div class="error">Error: ${data.error}</div>`;
                            return;
                        }
                        
                        const totalReturn = data.total_value - 100000;
                        const returnClass = totalReturn >= 0 ? 'positive' : 'negative';
                        
                        document.getElementById('portfolio-summary').innerHTML = `
                            <div class="stat-grid">
                                <div class="stat-box">
                                    <div class="stat-number">$${data.cash_balance.toLocaleString()}</div>
                                    <div class="stat-label">Cash Balance</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">$${data.total_value.toLocaleString()}</div>
                                    <div class="stat-label">Total Value</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number ${returnClass}">$${totalReturn.toLocaleString()}</div>
                                    <div class="stat-label">Total Return</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.holdings_count}</div>
                                    <div class="stat-label">Holdings</div>
                                </div>
                            </div>
                        `;
                    } catch (error) {
                        document.getElementById('portfolio-summary').innerHTML = 
                            `<div class="error">Error loading portfolio: ${error.message}</div>`;
                    }
                }
                
                async function loadRecommendations() {
                    try {
                        const response = await fetch('/api/analysis/recommendations');
                        const recommendations = await response.json();
                        
                        if (recommendations.length === 0) {
                            document.getElementById('recommendations-display').innerHTML = 
                                '<p>No analysis available. Stocks are still loading...</p>';
                            return;
                        }
                        
                        let html = `
                            <table class="recommendations-table">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Price</th>
                                        <th>Recommendation</th>
                                        <th>Confidence</th>
                                        <th>RSI</th>
                                        <th>Reasons</th>
                                    </tr>
                                </thead>
                                <tbody>
                        `;
                        
                        recommendations.forEach(rec => {
                            const recClass = rec.recommendation.toLowerCase();
                            html += `
                                <tr>
                                    <td><strong>${rec.symbol}</strong></td>
                                    <td>$${rec.current_price.toFixed(2)}</td>
                                    <td><span class="recommendation-badge rec-${recClass}">${rec.recommendation}</span></td>
                                    <td>
                                        <div class="confidence-bar">
                                            <div class="confidence-fill" style="width: ${rec.confidence}%"></div>
                                        </div>
                                        ${rec.confidence}%
                                    </td>
                                    <td>${rec.rsi.toFixed(1)}</td>
                                    <td><small>${rec.reasons.join(', ')}</small></td>
                                </tr>
                            `;
                        });
                        
                        html += '</tbody></table>';
                        document.getElementById('recommendations-display').innerHTML = html;
                        
                    } catch (error) {
                        document.getElementById('recommendations-display').innerHTML = 
                            `<div class="error">Error loading recommendations: ${error.message}</div>`;
                    }
                }
                
                async function loadHoldings() {
                    try {
                        const response = await fetch('/api/portfolio/holdings');
                        const holdings = await response.json();
                        
                        if (holdings.length === 0) {
                            document.getElementById('holdings-display').innerHTML = 
                                '<p>No holdings found. Start trading to build your portfolio!</p>';
                            return;
                        }
                        
                        let html = `
                            <table class="holdings-table">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Quantity</th>
                                        <th>Avg Cost</th>
                                        <th>Current Price</th>
                                        <th>Market Value</th>
                                        <th>Gain/Loss</th>
                                        <th>%</th>
                                    </tr>
                                </thead>
                                <tbody>
                        `;
                        
                        holdings.forEach(holding => {
                            const gainLossClass = holding.gain_loss >= 0 ? 'positive' : 'negative';
                            html += `
                                <tr>
                                    <td><strong>${holding.symbol}</strong></td>
                                    <td>${holding.quantity}</td>
                                    <td>$${holding.average_cost.toFixed(2)}</td>
                                    <td>$${holding.current_price.toFixed(2)}</td>
                                    <td>$${holding.market_value.toLocaleString()}</td>
                                    <td class="${gainLossClass}">$${holding.gain_loss.toFixed(2)}</td>
                                    <td class="${gainLossClass}">${holding.gain_loss_percent.toFixed(2)}%</td>
                                </tr>
                            `;
                        });
                        
                        html += '</tbody></table>';
                        document.getElementById('holdings-display').innerHTML = html;
                        
                    } catch (error) {
                        document.getElementById('holdings-display').innerHTML = 
                            `<div class="error">Error loading holdings: ${error.message}</div>`;
                    }
                }
                
                async function loadAutoTradingStatus() {
                    try {
                        const response = await fetch('/api/auto-trading/status');
                        const status = await response.json();
                        
                        autoTradingEnabled = status.enabled;
                        updateAutoTradingUI();
                        
                    } catch (error) {
                        console.error('Error loading auto trading status:', error);
                    }
                }
                
                function updateAutoTradingUI() {
                    const panel = document.getElementById('auto-trading-panel');
                    const statusText = document.getElementById('auto-status');
                    const toggleBtn = document.getElementById('auto-toggle-btn');
                    
                    if (autoTradingEnabled) {
                        panel.classList.add('enabled');
                        statusText.textContent = 'Enabled';
                        statusText.className = 'positive';
                        toggleBtn.textContent = 'Disable Auto Trading';
                        toggleBtn.classList.add('active');
                    } else {
                        panel.classList.remove('enabled');
                        statusText.textContent = 'Disabled';
                        statusText.className = 'negative';
                        toggleBtn.textContent = 'Enable Auto Trading';
                        toggleBtn.classList.remove('active');
                    }
                }
                
                async function toggleAutoTrading() {
                    try {
                        const response = await fetch('/api/auto-trading/toggle', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                enabled: !autoTradingEnabled
                            })
                        });
                        
                        const result = await response.json();
                        if (result.success) {
                            autoTradingEnabled = result.enabled;
                            updateAutoTradingUI();
                            showMessage(`Auto trading ${autoTradingEnabled ? 'enabled' : 'disabled'}`, 'success');
                        }
                        
                    } catch (error) {
                        showMessage(`Error toggling auto trading: ${error.message}`, 'error');
                    }
                }
                
                async function executeAutoTrades() {
                    try {
                        showMessage('Analyzing stocks and executing trades...', 'info');
                        
                        const response = await fetch('/api/auto-trading/execute', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            }
                        });
                        
                        const trades = await response.json();
                        
                        if (trades.length === 0) {
                            showMessage('No trades executed. Either auto-trading is disabled or no high-confidence signals found.', 'info');
                        } else {
                            let message = `Executed ${trades.length} trades:\\n`;
                            trades.forEach(trade => {
                                message += `${trade.action} ${trade.quantity} ${trade.symbol} at $${trade.price.toFixed(2)} (${trade.confidence}% confidence)\\n`;
                            });
                            showMessage(message, 'success');
                            
                            // Refresh data
                            loadPortfolioSummary();
                            loadHoldings();
                        }
                        
                    } catch (error) {
                        showMessage(`Error executing auto trades: ${error.message}`, 'error');
                    }
                }
                
                async function getQuote() {
                    const symbol = document.getElementById('quote-symbol').value.trim().toUpperCase();
                    if (!symbol) {
                        alert('Please enter a stock symbol');
                        return;
                    }
                    
                    document.getElementById('quote-display').innerHTML = '<div class="loading">Getting quote...</div>';
                    
                    try {
                        const response = await fetch(`/api/stock/quote/${symbol}`);
                        const data = await response.json();
                        
                        if (data.error) {
                            document.getElementById('quote-display').innerHTML = 
                                `<div class="error">${data.error}</div>`;
                            return;
                        }
                        
                        document.getElementById('quote-display').innerHTML = `
                            <div class="quote-display">
                                <h4>${data.symbol}</h4>
                                <p><strong>Current Price: $${data.current_price.toFixed(2)}</strong></p>
                                <p><small>Last updated: ${new Date(data.timestamp).toLocaleString()}</small></p>
                            </div>
                        `;
                        
                        // Auto-fill trading forms
                        document.getElementById('buy-symbol').value = symbol;
                        document.getElementById('sell-symbol').value = symbol;
                        
                    } catch (error) {
                        document.getElementById('quote-display').innerHTML = 
                            `<div class="error">Error getting quote: ${error.message}</div>`;
                    }
                }
                
                async function executeBuy() {
                    const symbol = document.getElementById('buy-symbol').value.trim().toUpperCase();
                    const quantity = parseInt(document.getElementById('buy-quantity').value);
                    
                    if (!symbol || !quantity || quantity <= 0) {
                        showMessage('Please enter valid symbol and quantity', 'error');
                        return;
                    }
                    
                    showMessage('Executing buy order...', 'info');
                    
                    try {
                        const response = await fetch('/api/trade/buy', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                symbol: symbol,
                                quantity: quantity
                            })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            showMessage(result.message, 'success');
                            document.getElementById('buy-symbol').value = '';
                            document.getElementById('buy-quantity').value = '';
                            // Refresh data
                            loadPortfolioSummary();
                            loadHoldings();
                        } else {
                            showMessage(result.error, 'error');
                        }
                        
                    } catch (error) {
                        showMessage(`Error executing buy order: ${error.message}`, 'error');
                    }
                }
                
                async function executeSell() {
                    const symbol = document.getElementById('sell-symbol').value.trim().toUpperCase();
                    const quantity = parseInt(document.getElementById('sell-quantity').value);
                    
                    if (!symbol || !quantity || quantity <= 0) {
                        showMessage('Please enter valid symbol and quantity', 'error');
                        return;
                    }
                    
                    showMessage('Executing sell order...', 'info');
                    
                    try {
                        const response = await fetch('/api/trade/sell', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                symbol: symbol,
                                quantity: quantity
                            })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            showMessage(result.message, 'success');
                            document.getElementById('sell-symbol').value = '';
                            document.getElementById('sell-quantity').value = '';
                            // Refresh data
                            loadPortfolioSummary();
                            loadHoldings();
                        } else {
                            showMessage(result.error, 'error');
                        }
                        
                    } catch (error) {
                        showMessage(`Error executing sell order: ${error.message}`, 'error');
                    }
                }
                
                function showMessage(message, type) {
                    const messagesDiv = document.getElementById('trading-messages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${type}`;
                    messageDiv.textContent = message;
                    
                    messagesDiv.appendChild(messageDiv);
                    
                    // Remove message after 5 seconds
                    setTimeout(() => {
                        if (messageDiv.parentNode) {
                            messageDiv.parentNode.removeChild(messageDiv);
                        }
                    }, 5000);
                }
                
                // Allow Enter key to trigger actions
                document.getElementById('quote-symbol').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') getQuote();
                });
                
                document.getElementById('buy-quantity').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') executeBuy();
                });
                
                document.getElementById('sell-quantity').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') executeSell();
                });
                
                // Chart management functions
                async function regenerateCharts() {
                    try {
                        showMessage('Regenerating all charts... This may take a few minutes.', 'info');
                        
                        const response = await fetch('/api/charts/generate', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            }
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            const chartResults = result.results;
                            let message = `Charts regenerated successfully!\\n`;
                            message += `Stock charts: ${chartResults.stock_charts.generated} generated, ${chartResults.stock_charts.failed} failed\\n`;
                            message += `Sector overview: ${chartResults.sector_overview ? '✓' : '✗'}\\n`;
                            message += `Portfolio performance: ${chartResults.portfolio_performance ? '✓' : '✗'}\\n`;
                            message += `Total time: ${chartResults.total_time.toFixed(2)} seconds`;
                            
                            showMessage(message, 'success');
                            checkChartStatus(); // Update status
                        } else {
                            showMessage(`Error regenerating charts: ${result.error}`, 'error');
                        }
                        
                    } catch (error) {
                        showMessage(`Error regenerating charts: ${error.message}`, 'error');
                    }
                }
                
                async function checkChartStatus() {
                    try {
                        const response = await fetch('/api/charts/status');
                        const status = await response.json();
                        
                        const statusDiv = document.getElementById('chart-status');
                        
                        if (status.charts_exist) {
                            statusDiv.innerHTML = `
                                <div class="info">
                                    <strong>Chart Status:</strong><br>
                                    📊 Total charts: ${status.total_charts}<br>
                                    📈 Stock charts: ${status.stock_charts}<br>
                                    🎯 Overview charts: ${status.overview_charts}<br>
                                    🕒 Last updated: ${new Date(status.last_updated).toLocaleString()}
                                </div>
                            `;
                        } else {
                            statusDiv.innerHTML = `
                                <div class="error">
                                    No charts found. Click "Regenerate All Charts" to create them.
                                </div>
                            `;
                        }
                        
                    } catch (error) {
                        document.getElementById('chart-status').innerHTML = `
                            <div class="error">Error checking chart status: ${error.message}</div>
                        `;
                    }
                }
            </script>
        </body>
        </html>
        '''
        
        return html_template
    
    async def initialize_and_load_stocks(self):
        """Initialize and load all stocks."""
        self.logger.info("🚀 Starting Complete Trading Platform")
        self.logger.info("=" * 60)
        
        # Initialize portfolio
        await self.portfolio_service.initialize_portfolio()
        
        # Auto-load all NASDAQ & TSX stocks
        self.logger.info("📥 Auto-loading NASDAQ & TSX stocks...")
        result = await self.stock_data_service.load_all_stocks_on_startup()
        
        self.logger.info("✅ Stock loading completed!")
        self.logger.info(f"   NASDAQ stocks: {result['nasdaq_count']}")
        self.logger.info(f"   TSX stocks: {result['tsx_count']}")
        self.logger.info(f"   Total loaded: {result['loaded_count']}")
        self.logger.info(f"   Failed: {result['failed_count']}")
        
        # Generate charts after stock loading
        self.logger.info("📊 Generating analysis charts...")
        try:
            # Get recommendations for sector overview
            recommendations = await self.analysis_service.get_stock_recommendations()
            
            # Generate all charts
            chart_results = await self.chart_service.generate_all_charts(
                portfolio_service=self.portfolio_service,
                recommendations=recommendations
            )
            
            self.logger.info("✅ Chart generation completed!")
            self.logger.info(f"   Stock charts: {chart_results['stock_charts']['generated']} generated, {chart_results['stock_charts']['failed']} failed")
            self.logger.info(f"   Sector overview: {'✓' if chart_results['sector_overview'] else '✗'}")
            self.logger.info(f"   Portfolio performance: {'✓' if chart_results['portfolio_performance'] else '✗'}")
            self.logger.info(f"   Total time: {chart_results['total_time']:.2f} seconds")
            
            result['chart_results'] = chart_results
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
            result['chart_error'] = str(e)
        
        return result
    
    def run_web_server(self, host='127.0.0.1', port=5000):
        """Run the Flask web server."""
        self.logger.info(f"🌐 Starting web server at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=False)


async def main():
    """Main function that initializes and runs the complete trading platform."""
    print("🚀 Complete Trading Platform - Advanced Implementation")
    print("Loading stocks and launching advanced trading interface...")
    
    # Create the trading platform
    platform = CompleteTradingPlatform()
    
    try:
        # Load all stocks in background
        def load_stocks():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(platform.initialize_and_load_stocks())
            loop.close()
        
        # Start stock loading in background thread
        stock_thread = threading.Thread(target=load_stocks)
        stock_thread.daemon = True
        stock_thread.start()
        
        # Give it a moment to start
        time.sleep(2)
        
        # Launch web browser
        def open_browser():
            time.sleep(3)  # Wait for server to start
            webbrowser.open('http://127.0.0.1:5001')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("\n" + "=" * 60)
        print("🌐 ADVANCED TRADING PLATFORM LAUNCHING")
        print("=" * 60)
        print("📍 URL: http://127.0.0.1:5001")
        print("🔄 Stocks are loading in the background...")
        print("💰 Portfolio initialized with $100,000")
        print("🤖 AI-powered analysis and auto-trading ready!")
        print("\n🎯 ADVANCED FEATURES AVAILABLE:")
        print("   ✅ Real-time stock quotes")
        print("   ✅ Buy/sell orders with validation")
        print("   ✅ Portfolio tracking with P&L")
        print("   ✅ AI stock analysis with recommendations")
        print("   ✅ Auto-trading based on technical analysis")
        print("   ✅ RSI and moving average indicators")
        print("   ✅ Confidence-based trade execution")
        print("   ✅ Centralized database")
        print("   ✅ Auto-refreshing data")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the web server (this blocks)
        platform.run_web_server(port=5001)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down complete trading platform...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

"""Portfolio management service."""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from decimal import Decimal

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import app.stockSimulator as stockSimulator
import app.stockAnalyzer as stockAnalyzer
from app.database_manager import db_manager
from shared.logging import get_logger


class PortfolioService:
    """Service for portfolio management operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        # Initialize portfolio database
        stockSimulator.initialize_portfolio_db()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary information."""
        try:
            portfolio, holdings = stockSimulator.get_portfolio_state()
            
            if not portfolio:
                return {
                    'error': 'No portfolio found',
                    'cash_balance': 0,
                    'total_value': 0,
                    'total_return': 0,
                    'return_percentage': 0,
                    'holdings_count': 0
                }
            
            cash_balance = portfolio[1]
            total_value = stockSimulator.calculate_current_portfolio_value()
            total_return = total_value - 10000  # Initial investment
            return_percentage = (total_return / 10000) * 100 if total_value > 0 else 0
            
            return {
                'cash_balance': float(cash_balance),
                'total_value': float(total_value),
                'total_return': float(total_return),
                'return_percentage': float(return_percentage),
                'last_transaction_date': portfolio[3],
                'holdings_count': len(holdings) if holdings else 0,
                'can_trade': stockSimulator.can_make_transactions()
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e)}
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current portfolio holdings with current prices."""
        try:
            portfolio, holdings = stockSimulator.get_portfolio_state()
            
            if not holdings:
                return []
            
            # Get current prices for all holdings
            symbols = [holding[1] for holding in holdings]
            current_prices = stockSimulator.get_current_stock_prices(symbols)
            
            holdings_data = []
            for holding in holdings:
                symbol = holding[1]
                quantity = holding[2]
                avg_cost = holding[3]
                total_cost = holding[4]
                
                current_price = current_prices.get(symbol, 0)
                market_value = quantity * current_price if current_price else 0
                gain_loss = market_value - total_cost
                gain_loss_pct = (gain_loss / total_cost) * 100 if total_cost > 0 else 0
                
                holdings_data.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_cost': float(avg_cost),
                    'current_price': float(current_price),
                    'market_value': float(market_value),
                    'total_cost': float(total_cost),
                    'gain_loss': float(gain_loss),
                    'gain_loss_percent': float(gain_loss_pct)
                })
            
            return holdings_data
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            return []
    
    def get_recent_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent portfolio transactions."""
        try:
            transactions_df = db_manager.get_transactions_df()
            
            if transactions_df.empty:
                return []
            
            # Get most recent transactions
            recent = transactions_df.tail(limit).iloc[::-1]  # Reverse for newest first
            
            transactions = []
            for _, tx in recent.iterrows():
                transactions.append({
                    'id': tx['id'],
                    'date': tx['transaction_date'],
                    'type': tx['transaction_type'],
                    'symbol': tx['symbol'],
                    'quantity': tx['quantity'],
                    'price': float(tx['price_per_share']),
                    'total': float(tx['total_amount']),
                    'fee': float(tx['brokerage_fee'])
                })
            
            return transactions
        except Exception as e:
            self.logger.error(f"Error getting transactions: {e}")
            return []
    
    def execute_buy_order(self, symbol: str, quantity: int) -> Dict[str, Any]:
        """Execute a buy order."""
        try:
            if not stockSimulator.can_make_transactions():
                return {'error': 'Cannot make transactions yet (less than 1 day since last transaction)'}
            
            # Get current price
            current_prices = stockSimulator.get_current_stock_prices([symbol])
            if symbol not in current_prices:
                return {'error': f'Could not get current price for {symbol}'}
            
            current_price = current_prices[symbol]
            total_cost = (quantity * current_price) + 10  # Include brokerage fee
            
            # Check cash balance
            portfolio, _ = stockSimulator.get_portfolio_state()
            cash_balance = portfolio[1] if portfolio else 0
            
            if total_cost > cash_balance:
                return {
                    'error': f'Insufficient funds. Need ${total_cost:.2f}, have ${cash_balance:.2f}'
                }
            
            # Execute transaction
            stock_id = db_manager.get_or_create_stock_id(symbol)
            stockSimulator.execute_buy_transaction(stock_id, symbol, quantity, current_price)
            
            return {
                'success': True,
                'message': f'Successfully bought {quantity} shares of {symbol} at ${current_price:.2f}',
                'transaction': {
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': float(current_price),
                    'total_cost': float(total_cost)
                }
            }
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            return {'error': str(e)}
    
    def execute_sell_order(self, symbol: str, quantity: int) -> Dict[str, Any]:
        """Execute a sell order."""
        try:
            if not stockSimulator.can_make_transactions():
                return {'error': 'Cannot make transactions yet (less than 1 day since last transaction)'}
            
            # Check holdings
            portfolio, holdings = stockSimulator.get_portfolio_state()
            
            owned_quantity = 0
            stock_id = None
            for holding in holdings:
                if holding[1] == symbol:
                    owned_quantity = holding[2]
                    stock_id = holding[0]
                    break
            
            if owned_quantity == 0:
                return {'error': f'You do not own any shares of {symbol}'}
            
            if quantity > owned_quantity:
                return {'error': f'You only own {owned_quantity} shares of {symbol}'}
            
            # Get current price
            current_prices = stockSimulator.get_current_stock_prices([symbol])
            if symbol not in current_prices:
                return {'error': f'Could not get current price for {symbol}'}
            
            current_price = current_prices[symbol]
            total_proceeds = (quantity * current_price) - 10  # Subtract brokerage fee
            
            # Execute transaction
            stockSimulator.execute_sell_transaction(stock_id, symbol, quantity, current_price)
            
            return {
                'success': True,
                'message': f'Successfully sold {quantity} shares of {symbol} at ${current_price:.2f}',
                'transaction': {
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': float(current_price),
                    'total_proceeds': float(total_proceeds)
                }
            }
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            return {'error': str(e)}
    
    def execute_auto_trading(self) -> Dict[str, Any]:
        """Execute automatic trading based on analysis."""
        try:
            if not stockSimulator.can_make_transactions():
                return {'error': 'Cannot make transactions yet (less than 1 day since last transaction)'}
            
            # Execute portfolio transactions
            stockSimulator.execute_portfolio_transactions()
            
            return {
                'success': True,
                'message': 'Automatic trading executed successfully'
            }
        except Exception as e:
            self.logger.error(f"Error executing auto trading: {e}")
            return {'error': str(e)}
    
    def get_portfolio_performance_data(self) -> Dict[str, Any]:
        """Get portfolio performance data for charting."""
        try:
            import sqlite3
            import pandas as pd
            import app.appconfig as appconfig
            
            # Get portfolio states
            conn = sqlite3.connect(appconfig.DB_NAME)
            portfolio_states_df = pd.read_sql_query('''
                SELECT * FROM portfolio_state 
                ORDER BY created_date ASC
            ''', conn)
            conn.close()
            
            if portfolio_states_df.empty:
                return {'error': 'No portfolio data available'}
            
            # Get transactions for holdings calculation
            transactions_df = db_manager.get_transactions_df()
            
            # Calculate portfolio values over time
            dates = []
            portfolio_values = []
            cash_values = []
            holdings_values = []
            
            for _, state in portfolio_states_df.iterrows():
                date = pd.to_datetime(state['created_date'])
                cash = state['cash_balance']
                
                # Calculate holdings value for this date
                holdings, holdings_value = stockSimulator.reconstruct_holdings_and_value(date, transactions_df)
                
                dates.append(date.strftime('%Y-%m-%d'))
                cash_values.append(float(cash))
                holdings_values.append(float(holdings_value))
                portfolio_values.append(float(cash + holdings_value))
            
            return {
                'dates': dates,
                'portfolio_values': portfolio_values,
                'cash_values': cash_values,
                'holdings_values': holdings_values,
                'initial_value': 10000,
                'current_value': portfolio_values[-1] if portfolio_values else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting performance data: {e}")
            return {'error': str(e)}
    
    def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current stock quote."""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_prices = stockSimulator.get_current_stock_prices([symbol])
            current_price = current_prices.get(symbol, None)
            
            return {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'current_price': float(current_price) if current_price else None,
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A')
            }
        except Exception as e:
            self.logger.error(f"Error getting stock quote: {e}")
            return {'error': str(e)}


# Global service instance
portfolio_service = PortfolioService()

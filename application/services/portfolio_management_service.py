"""Application service for portfolio management operations."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from ...domain.entities.portfolio import Portfolio, Holding, Transaction, TransactionType
from ...domain.repositories.portfolio_repository import IPortfolioRepository
from ...domain.repositories.stock_repository import IStockRepository
from ...application.services.stock_data_service import StockDataService
from ...shared.logging import get_logger


class PortfolioManagementService:
    """Application service for portfolio management operations."""
    
    def __init__(
        self, 
        portfolio_repository: IPortfolioRepository,
        stock_repository: IStockRepository,
        stock_data_service: StockDataService
    ):
        self.portfolio_repo = portfolio_repository
        self.stock_repo = stock_repository
        self.stock_data_service = stock_data_service
        self.logger = get_logger(__name__)
        
        # Trading parameters
        self.initial_cash = Decimal('10000')
        self.transaction_fee = Decimal('10')
        self.max_position_size = Decimal('0.1')  # 10% max per position
    
    async def initialize_portfolio(self, name: str = "Default Portfolio") -> Portfolio:
        """Initialize a new portfolio."""
        portfolio = Portfolio(
            id=None,
            name=name,
            initial_cash=self.initial_cash,
            current_cash=self.initial_cash,
            created_date=datetime.now().date(),
            updated_date=datetime.now().date()
        )
        
        return await self.portfolio_repo.create_portfolio(portfolio)
    
    async def get_or_create_default_portfolio(self) -> Portfolio:
        """Get the default portfolio or create it if it doesn't exist."""
        # For simplicity, we'll use portfolio ID 1 as default
        portfolio = await self.portfolio_repo.get_portfolio_by_id(1)
        if not portfolio:
            portfolio = await self.initialize_portfolio()
        return portfolio
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary information."""
        try:
            portfolio = await self.get_or_create_default_portfolio()
            holdings = await self.portfolio_repo.get_holdings(portfolio.id)
            
            # Calculate current portfolio value
            total_holdings_value = Decimal('0')
            for holding in holdings:
                current_price = await self.stock_data_service.get_current_price(holding.symbol)
                if current_price:
                    total_holdings_value += holding.quantity * current_price
                else:
                    # Fallback to cost basis if no current price
                    total_holdings_value += holding.avg_cost * holding.quantity
            
            current_value = portfolio.current_cash + total_holdings_value
            total_return = current_value - portfolio.initial_cash
            return_percentage = (total_return / portfolio.initial_cash) * 100 if portfolio.initial_cash > 0 else 0
            
            # Check if trading is allowed
            last_transaction_date = await self.portfolio_repo.get_last_transaction_date(portfolio.id)
            can_trade = True
            if last_transaction_date:
                # Allow trading if more than 1 hour has passed (for demo purposes)
                can_trade = (datetime.now() - last_transaction_date).total_seconds() > 3600
            
            return {
                'cash_balance': float(portfolio.current_cash),
                'current_value': float(current_value),
                'total_return': float(total_return),
                'return_percentage': float(return_percentage),
                'holdings_count': len(holdings),
                'can_trade': can_trade,
                'last_transaction_date': last_transaction_date.isoformat() if last_transaction_date else None,
                'initial_value': float(portfolio.initial_cash)
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e)}
    
    async def get_holdings_with_current_values(self) -> List[Dict[str, Any]]:
        """Get current holdings with market values."""
        try:
            portfolio = await self.get_or_create_default_portfolio()
            holdings = await self.portfolio_repo.get_holdings(portfolio.id)
            
            holdings_data = []
            for holding in holdings:
                current_price = await self.stock_data_service.get_current_price(holding.symbol)
                if current_price is None:
                    current_price = holding.avg_cost  # Fallback to cost basis
                
                market_value = holding.quantity * current_price
                total_cost = holding.quantity * holding.avg_cost
                gain_loss = market_value - total_cost
                gain_loss_percent = (gain_loss / total_cost) * 100 if total_cost > 0 else 0
                
                holdings_data.append({
                    'symbol': holding.symbol,
                    'quantity': holding.quantity,
                    'avg_cost': float(holding.avg_cost),
                    'current_price': float(current_price),
                    'market_value': float(market_value),
                    'total_cost': float(total_cost),
                    'gain_loss': float(gain_loss),
                    'gain_loss_percent': float(gain_loss_percent)
                })
            
            return holdings_data
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            return []
    
    async def execute_buy_order(self, symbol: str, quantity: int) -> Dict[str, Any]:
        """Execute a buy order."""
        try:
            portfolio = await self.get_or_create_default_portfolio()
            
            # Check if trading is allowed
            last_transaction_date = await self.portfolio_repo.get_last_transaction_date(portfolio.id)
            if last_transaction_date:
                time_since_last = (datetime.now() - last_transaction_date).total_seconds()
                if time_since_last < 3600:  # 1 hour cooldown for demo
                    return {'error': 'Trading cooldown active. Please wait before making another trade.'}
            
            # Get current price
            current_price = await self.stock_data_service.get_current_price(symbol)
            if current_price is None:
                return {'error': f'Could not get current price for {symbol}'}
            
            total_cost = (quantity * current_price) + self.transaction_fee
            
            if total_cost > portfolio.current_cash:
                return {
                    'error': f'Insufficient funds: need ${total_cost:.2f}, have ${portfolio.current_cash:.2f}'
                }
            
            # Get or create stock
            stock = await self.stock_repo.get_stock_by_symbol(symbol)
            if not stock:
                exchange = "TSX" if ".TO" in symbol else "NASDAQ"
                stock = await self.stock_repo.create_stock({
                    'symbol': symbol,
                    'name': f"{symbol} Inc.",
                    'exchange': exchange
                })
            
            # Create transaction
            transaction = Transaction(
                id=None,
                portfolio_id=portfolio.id,
                symbol=symbol,
                transaction_type=TransactionType.BUY,
                quantity=quantity,
                price=current_price,
                total_amount=quantity * current_price,
                fee=self.transaction_fee,
                transaction_date=datetime.now()
            )
            
            await self.portfolio_repo.create_transaction(transaction)
            
            # Update or create holding
            existing_holdings = await self.portfolio_repo.get_holdings(portfolio.id)
            existing_holding = next((h for h in existing_holdings if h.symbol == symbol), None)
            
            if existing_holding:
                # Update existing holding
                new_quantity = existing_holding.quantity + quantity
                new_total_cost = (existing_holding.quantity * existing_holding.avg_cost) + (quantity * current_price)
                new_avg_cost = new_total_cost / new_quantity
                
                updated_holding = Holding(
                    id=existing_holding.id,
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    quantity=new_quantity,
                    avg_cost=new_avg_cost,
                    created_date=existing_holding.created_date,
                    updated_date=datetime.now().date()
                )
                await self.portfolio_repo.create_or_update_holding(updated_holding)
            else:
                # Create new holding
                new_holding = Holding(
                    id=None,
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=current_price,
                    created_date=datetime.now().date(),
                    updated_date=datetime.now().date()
                )
                await self.portfolio_repo.create_or_update_holding(new_holding)
            
            # Update portfolio cash
            portfolio.current_cash -= total_cost
            portfolio.updated_date = datetime.now().date()
            await self.portfolio_repo.update_portfolio(portfolio)
            
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
    
    async def execute_sell_order(self, symbol: str, quantity: int) -> Dict[str, Any]:
        """Execute a sell order."""
        try:
            portfolio = await self.get_or_create_default_portfolio()
            
            # Check if trading is allowed
            last_transaction_date = await self.portfolio_repo.get_last_transaction_date(portfolio.id)
            if last_transaction_date:
                time_since_last = (datetime.now() - last_transaction_date).total_seconds()
                if time_since_last < 3600:  # 1 hour cooldown for demo
                    return {'error': 'Trading cooldown active. Please wait before making another trade.'}
            
            # Check holdings
            holdings = await self.portfolio_repo.get_holdings(portfolio.id)
            holding = next((h for h in holdings if h.symbol == symbol), None)
            
            if not holding or holding.quantity < quantity:
                owned = holding.quantity if holding else 0
                return {'error': f'Insufficient shares: need {quantity}, have {owned}'}
            
            # Get current price
            current_price = await self.stock_data_service.get_current_price(symbol)
            if current_price is None:
                return {'error': f'Could not get current price for {symbol}'}
            
            gross_proceeds = quantity * current_price
            net_proceeds = gross_proceeds - self.transaction_fee
            
            # Create transaction
            transaction = Transaction(
                id=None,
                portfolio_id=portfolio.id,
                symbol=symbol,
                transaction_type=TransactionType.SELL,
                quantity=quantity,
                price=current_price,
                total_amount=gross_proceeds,
                fee=self.transaction_fee,
                transaction_date=datetime.now()
            )
            
            await self.portfolio_repo.create_transaction(transaction)
            
            # Update holding
            new_quantity = holding.quantity - quantity
            if new_quantity == 0:
                await self.portfolio_repo.delete_holding(portfolio.id, symbol)
            else:
                updated_holding = Holding(
                    id=holding.id,
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    quantity=new_quantity,
                    avg_cost=holding.avg_cost,  # Keep same average cost
                    created_date=holding.created_date,
                    updated_date=datetime.now().date()
                )
                await self.portfolio_repo.create_or_update_holding(updated_holding)
            
            # Update portfolio cash
            portfolio.current_cash += net_proceeds
            portfolio.updated_date = datetime.now().date()
            await self.portfolio_repo.update_portfolio(portfolio)
            
            return {
                'success': True,
                'message': f'Successfully sold {quantity} shares of {symbol} at ${current_price:.2f}',
                'transaction': {
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': float(current_price),
                    'net_proceeds': float(net_proceeds)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            return {'error': str(e)}
    
    async def get_recent_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent transactions."""
        try:
            portfolio = await self.get_or_create_default_portfolio()
            transactions = await self.portfolio_repo.get_transactions(portfolio.id, limit=limit)
            
            return [
                {
                    'id': tx.id,
                    'date': tx.transaction_date.isoformat() if tx.transaction_date else None,
                    'type': tx.transaction_type.value,
                    'symbol': tx.symbol,
                    'quantity': tx.quantity,
                    'price': float(tx.price),
                    'total': float(tx.total_amount),
                    'fee': float(tx.fee)
                }
                for tx in transactions
            ]
        except Exception as e:
            self.logger.error(f"Error getting transactions: {e}")
            return []

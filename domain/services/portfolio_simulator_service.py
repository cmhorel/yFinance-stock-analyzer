"""Portfolio simulator service using clean architecture."""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
import random

from ..entities.stock import Stock, StockPrice
from ..entities.portfolio import Portfolio, PortfolioHolding, Transaction, TransactionType
from ..entities.analysis import AnalysisResult
from ..repositories.stock_repository import IStockRepository
from ..repositories.portfolio_repository import IPortfolioRepository
from .stock_analysis_service import StockAnalysisService
from shared.logging import get_logger


class PortfolioSimulatorService:
    """Service for simulating portfolio trading using analysis results."""
    
    def __init__(
        self, 
        stock_repo: IStockRepository,
        portfolio_repo: IPortfolioRepository,
        initial_cash: Decimal = Decimal('10000')
    ):
        self.stock_repo = stock_repo
        self.portfolio_repo = portfolio_repo
        self.analysis_service = StockAnalysisService()
        self.initial_cash = initial_cash
        self.logger = get_logger(__name__)
        
        # Trading parameters
        self.max_position_size = Decimal('0.1')  # Max 10% of portfolio per position
        self.min_trade_amount = Decimal('100')   # Minimum trade amount
        self.transaction_fee = Decimal('10')     # Fixed transaction fee
        self.risk_tolerance = Decimal('0.3')     # Maximum volatility tolerance
        
    async def initialize_portfolio(self, name: str = "Simulation Portfolio") -> Portfolio:
        """Initialize a new portfolio for simulation."""
        portfolio = Portfolio(
            id=None,
            name=name,
            cash_balance=self.initial_cash,
            total_portfolio_value=self.initial_cash,
            created_date=datetime.now(),
            updated_date=datetime.now()
        )
        
        return await self.portfolio_repo.create_portfolio_state(portfolio)
    
    async def get_current_portfolio(self) -> Optional[Portfolio]:
        """Get the current portfolio state."""
        return await self.portfolio_repo.get_current_portfolio()
    
    async def analyze_stocks_for_trading(self, symbols: List[str]) -> List[AnalysisResult]:
        """Analyze a list of stocks and return analysis results."""
        analysis_results = []
        
        for symbol in symbols:
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
        
        return analysis_results
    
    async def execute_buy_order(
        self, 
        portfolio: Portfolio, 
        symbol: str, 
        quantity: int, 
        price: Decimal
    ) -> Optional[Transaction]:
        """Execute a buy order."""
        try:
            # Get stock
            stock = await self.stock_repo.get_stock_by_symbol(symbol)
            if not stock:
                self.logger.error(f"Stock {symbol} not found")
                return None
            
            # Calculate costs
            total_cost = (quantity * price) + self.transaction_fee
            
            # Check if we have enough cash
            if total_cost > portfolio.cash_balance:
                self.logger.warning(f"Insufficient funds for {symbol}: need {total_cost}, have {portfolio.cash_balance}")
                return None
            
            # Create transaction
            transaction = Transaction(
                id=None,
                stock_id=stock.id,
                symbol=symbol,
                transaction_type=TransactionType.BUY,
                quantity=quantity,
                price_per_share=price,
                total_amount=total_cost,
                brokerage_fee=self.transaction_fee,
                transaction_date=datetime.now(),
                created_at=datetime.now()
            )
            
            # Save transaction
            saved_transaction = await self.portfolio_repo.create_transaction(transaction)
            
            # Update portfolio cash
            portfolio.cash_balance -= total_cost
            portfolio.updated_date = datetime.now()
            await self.portfolio_repo.update_portfolio_state(portfolio)
            
            # Update or create holding
            existing_holding = await self.portfolio_repo.get_holding_by_stock_id(stock.id)
            if existing_holding:
                # Update existing holding
                total_shares = existing_holding.quantity + quantity
                total_cost_basis = existing_holding.total_cost + (quantity * price)
                new_avg_cost = total_cost_basis / total_shares
                
                existing_holding.quantity = total_shares
                existing_holding.avg_cost_per_share = new_avg_cost
                existing_holding.total_cost = total_cost_basis
                
                await self.portfolio_repo.create_or_update_holding(existing_holding)
            else:
                # Create new holding
                new_holding = PortfolioHolding(
                    id=None,
                    stock_id=stock.id,
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost_per_share=price,
                    total_cost=quantity * price,
                    last_updated=datetime.now()
                )
                
                await self.portfolio_repo.create_or_update_holding(new_holding)
            
            self.logger.info(f"Executed BUY: {quantity} shares of {symbol} at ${price}")
            return saved_transaction
            
        except Exception as e:
            self.logger.error(f"Error executing buy order for {symbol}: {e}")
            return None
    
    async def execute_sell_order(
        self, 
        portfolio: Portfolio, 
        symbol: str, 
        quantity: int, 
        price: Decimal
    ) -> Optional[Transaction]:
        """Execute a sell order."""
        try:
            # Get stock
            stock = await self.stock_repo.get_stock_by_symbol(symbol)
            if not stock:
                self.logger.error(f"Stock {symbol} not found")
                return None
            
            # Get current holding
            holding = await self.portfolio_repo.get_holding_by_stock_id(stock.id)
            if not holding or holding.quantity < quantity:
                self.logger.warning(f"Insufficient shares for {symbol}: need {quantity}, have {holding.quantity if holding else 0}")
                return None
            
            # Calculate proceeds
            gross_proceeds = quantity * price
            net_proceeds = gross_proceeds - self.transaction_fee
            
            # Create transaction
            transaction = Transaction(
                id=None,
                stock_id=stock.id,
                symbol=symbol,
                transaction_type=TransactionType.SELL,
                quantity=quantity,
                price_per_share=price,
                total_amount=gross_proceeds,
                brokerage_fee=self.transaction_fee,
                transaction_date=datetime.now(),
                created_at=datetime.now()
            )
            
            # Save transaction
            saved_transaction = await self.portfolio_repo.create_transaction(transaction)
            
            # Update portfolio cash
            portfolio.cash_balance += net_proceeds
            portfolio.updated_date = datetime.now()
            await self.portfolio_repo.update_portfolio_state(portfolio)
            
            # Update holding
            holding.quantity -= quantity
            if holding.quantity == 0:
                # Remove holding if no shares left
                await self.portfolio_repo.delete_holding(stock.id)
            else:
                # Update holding
                holding.total_cost = holding.quantity * holding.avg_cost_per_share
                await self.portfolio_repo.create_or_update_holding(holding)
            
            self.logger.info(f"Executed SELL: {quantity} shares of {symbol} at ${price}")
            return saved_transaction
            
        except Exception as e:
            self.logger.error(f"Error executing sell order for {symbol}: {e}")
            return None
    
    async def simulate_trading_day(self, symbols: List[str]) -> Dict[str, Any]:
        """Simulate one day of trading based on analysis."""
        results = {
            "date": datetime.now().date(),
            "trades_executed": [],
            "analysis_results": [],
            "portfolio_value_before": Decimal('0'),
            "portfolio_value_after": Decimal('0'),
            "cash_before": Decimal('0'),
            "cash_after": Decimal('0')
        }
        
        try:
            # Get current portfolio
            portfolio = await self.get_current_portfolio()
            if not portfolio:
                portfolio = await self.initialize_portfolio()
            
            results["cash_before"] = portfolio.cash_balance
            results["portfolio_value_before"] = await self.calculate_portfolio_value(portfolio)
            
            # Analyze stocks
            analysis_results = await self.analyze_stocks_for_trading(symbols)
            results["analysis_results"] = len(analysis_results)
            
            if not analysis_results:
                self.logger.warning("No analysis results available for trading")
                return results
            
            # Get recommendations
            recommendations = self.analysis_service.get_recommendations(analysis_results)
            
            # Execute buy orders
            buy_recommendations = recommendations.get("buy", [])[:5]  # Top 5 buy signals
            for rec in buy_recommendations:
                if await self._should_buy(portfolio, rec):
                    quantity = await self._calculate_buy_quantity(portfolio, rec)
                    if quantity > 0:
                        transaction = await self.execute_buy_order(
                            portfolio, 
                            rec["symbol"], 
                            quantity, 
                            Decimal(str(rec["price"]))
                        )
                        if transaction:
                            results["trades_executed"].append({
                                "type": "BUY",
                                "symbol": rec["symbol"],
                                "quantity": quantity,
                                "price": rec["price"],
                                "total": float(transaction.total_amount)
                            })
            
            # Execute sell orders
            current_holdings = await self.portfolio_repo.get_current_holdings()
            sell_recommendations = recommendations.get("sell", [])
            
            for rec in sell_recommendations:
                # Check if we own this stock
                holding = next((h for h in current_holdings if h.symbol == rec["symbol"]), None)
                if holding and await self._should_sell(holding, rec):
                    quantity = await self._calculate_sell_quantity(holding, rec)
                    if quantity > 0:
                        transaction = await self.execute_sell_order(
                            portfolio, 
                            rec["symbol"], 
                            quantity, 
                            Decimal(str(rec["price"]))
                        )
                        if transaction:
                            results["trades_executed"].append({
                                "type": "SELL",
                                "symbol": rec["symbol"],
                                "quantity": quantity,
                                "price": rec["price"],
                                "total": float(transaction.total_amount)
                            })
            
            # Update final portfolio values
            portfolio = await self.get_current_portfolio()
            results["cash_after"] = portfolio.cash_balance
            results["portfolio_value_after"] = await self.calculate_portfolio_value(portfolio)
            
            self.logger.info(f"Trading day complete: {len(results['trades_executed'])} trades executed")
            
        except Exception as e:
            self.logger.error(f"Error in trading simulation: {e}")
            results["error"] = str(e)
        
        return results
    
    async def calculate_portfolio_value(self, portfolio: Portfolio) -> Decimal:
        """Calculate total portfolio value including holdings."""
        total_value = portfolio.cash_balance
        
        try:
            holdings = await self.portfolio_repo.get_current_holdings()
            
            for holding in holdings:
                # Get latest price
                latest_prices = await self.stock_repo.get_stock_prices(holding.stock_id, limit=1)
                if latest_prices:
                    current_price = latest_prices[0].close_price
                    holding_value = holding.quantity * current_price
                    total_value += holding_value
                    
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
        
        return total_value
    
    async def _should_buy(self, portfolio: Portfolio, recommendation: Dict[str, Any]) -> bool:
        """Determine if we should execute a buy order."""
        # Check risk tolerance
        if recommendation["risk_score"] > float(self.risk_tolerance):
            return False
        
        # Check confidence threshold
        if recommendation["confidence"] < 0.6:
            return False
        
        # Check if we already have a position
        try:
            holding = await self.portfolio_repo.get_holding_by_symbol(recommendation["symbol"])
            if holding:
                # Don't add to existing positions for now
                return False
        except:
            pass
        
        return True
    
    async def _should_sell(self, holding: PortfolioHolding, recommendation: Dict[str, Any]) -> bool:
        """Determine if we should execute a sell order."""
        # Check confidence threshold
        if recommendation["confidence"] < 0.7:
            return False
        
        # Check if we're at a loss (simple stop-loss)
        current_price = Decimal(str(recommendation["price"]))
        if current_price < holding.avg_cost_per_share * Decimal('0.9'):  # 10% stop loss
            return True
        
        return True
    
    async def _calculate_buy_quantity(self, portfolio: Portfolio, recommendation: Dict[str, Any]) -> int:
        """Calculate how many shares to buy."""
        try:
            price = Decimal(str(recommendation["price"]))
            
            # Calculate maximum position size (10% of portfolio)
            portfolio_value = await self.calculate_portfolio_value(portfolio)
            max_position_value = portfolio_value * self.max_position_size
            
            # Calculate affordable quantity
            available_cash = portfolio.cash_balance - self.transaction_fee
            max_affordable = int(available_cash / price)
            max_by_position = int(max_position_value / price)
            
            # Take the smaller of the two
            quantity = min(max_affordable, max_by_position)
            
            # Ensure minimum trade amount
            if quantity * price < self.min_trade_amount:
                return 0
            
            return max(0, quantity)
            
        except Exception as e:
            self.logger.error(f"Error calculating buy quantity: {e}")
            return 0
    
    async def _calculate_sell_quantity(self, holding: PortfolioHolding, recommendation: Dict[str, Any]) -> int:
        """Calculate how many shares to sell."""
        try:
            # For now, sell all shares when we get a sell signal
            return holding.quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating sell quantity: {e}")
            return 0
    
    async def get_portfolio_performance(self) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        try:
            portfolio = await self.get_current_portfolio()
            if not portfolio:
                return {"error": "No portfolio found"}
            
            current_value = await self.calculate_portfolio_value(portfolio)
            total_return = current_value - self.initial_cash
            return_percentage = (total_return / self.initial_cash) * 100
            
            holdings = await self.portfolio_repo.get_current_holdings()
            transactions = await self.portfolio_repo.get_transactions(limit=10)
            
            return {
                "initial_value": float(self.initial_cash),
                "current_value": float(current_value),
                "cash_balance": float(portfolio.cash_balance),
                "total_return": float(total_return),
                "return_percentage": float(return_percentage),
                "holdings_count": len(holdings),
                "total_transactions": len(transactions),
                "created_date": portfolio.created_date.isoformat() if portfolio.created_date else None,
                "last_updated": portfolio.updated_date.isoformat() if portfolio.updated_date else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio performance: {e}")
            return {"error": str(e)}

"""Portfolio repository interface."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from decimal import Decimal

from ..entities.portfolio import Portfolio, PortfolioHolding, Transaction, TransactionType


class IPortfolioRepository(ABC):
    """Interface for portfolio data repository operations."""
    
    # Portfolio Operations
    @abstractmethod
    async def get_portfolio_by_id(self, portfolio_id: int) -> Optional[Portfolio]:
        """Get a portfolio by its ID."""
        pass
    
    @abstractmethod
    async def get_portfolio_by_name(self, name: str) -> Optional[Portfolio]:
        """Get a portfolio by its name."""
        pass
    
    @abstractmethod
    async def get_all_portfolios(self) -> List[Portfolio]:
        """Get all portfolios."""
        pass
    
    @abstractmethod
    async def create_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Create a new portfolio."""
        pass
    
    @abstractmethod
    async def update_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Update an existing portfolio."""
        pass
    
    @abstractmethod
    async def delete_portfolio(self, portfolio_id: int) -> bool:
        """Delete a portfolio."""
        pass
    
    @abstractmethod
    async def get_default_portfolio(self) -> Optional[Portfolio]:
        """Get the default portfolio (if any)."""
        pass
    
    # Portfolio Holdings Operations
    @abstractmethod
    async def get_portfolio_holdings(self, portfolio_id: int) -> List[PortfolioHolding]:
        """Get all holdings for a portfolio."""
        pass
    
    @abstractmethod
    async def get_holding_by_stock(self, portfolio_id: int, stock_id: int) -> Optional[PortfolioHolding]:
        """Get a specific holding by stock ID."""
        pass
    
    @abstractmethod
    async def create_holding(self, holding: PortfolioHolding) -> PortfolioHolding:
        """Create a new portfolio holding."""
        pass
    
    @abstractmethod
    async def update_holding(self, holding: PortfolioHolding) -> PortfolioHolding:
        """Update an existing portfolio holding."""
        pass
    
    @abstractmethod
    async def delete_holding(self, holding_id: int) -> bool:
        """Delete a portfolio holding."""
        pass
    
    @abstractmethod
    async def update_holding_prices(self, price_updates: Dict[int, Decimal]) -> int:
        """Update current prices for multiple holdings. Returns count of updated holdings."""
        pass
    
    # Transaction Operations
    @abstractmethod
    async def get_transaction_by_id(self, transaction_id: int) -> Optional[Transaction]:
        """Get a transaction by its ID."""
        pass
    
    @abstractmethod
    async def get_portfolio_transactions(
        self, 
        portfolio_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        transaction_type: Optional[TransactionType] = None,
        stock_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Transaction]:
        """Get transactions for a portfolio with optional filtering."""
        pass
    
    @abstractmethod
    async def get_stock_transactions(
        self, 
        stock_id: int,
        portfolio_id: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[Transaction]:
        """Get all transactions for a specific stock."""
        pass
    
    @abstractmethod
    async def create_transaction(self, transaction: Transaction) -> Transaction:
        """Create a new transaction."""
        pass
    
    @abstractmethod
    async def update_transaction(self, transaction: Transaction) -> Transaction:
        """Update an existing transaction."""
        pass
    
    @abstractmethod
    async def delete_transaction(self, transaction_id: int) -> bool:
        """Delete a transaction."""
        pass
    
    @abstractmethod
    async def get_recent_transactions(
        self, 
        portfolio_id: Optional[int] = None,
        days_back: int = 30,
        limit: Optional[int] = None
    ) -> List[Transaction]:
        """Get recent transactions across portfolios or for a specific portfolio."""
        pass
    
    # Portfolio Analytics
    @abstractmethod
    async def get_portfolio_performance(
        self, 
        portfolio_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        pass
    
    @abstractmethod
    async def get_portfolio_allocation(self, portfolio_id: int) -> Dict[str, Decimal]:
        """Get current asset allocation by stock symbol."""
        pass
    
    @abstractmethod
    async def get_portfolio_sector_allocation(self, portfolio_id: int) -> Dict[str, Decimal]:
        """Get portfolio allocation by sector."""
        pass
    
    @abstractmethod
    async def get_portfolio_gains_losses(
        self, 
        portfolio_id: int,
        realized_only: bool = False
    ) -> Dict[str, Any]:
        """Get realized and/or unrealized gains and losses."""
        pass
    
    @abstractmethod
    async def get_portfolio_dividend_history(
        self, 
        portfolio_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Transaction]:
        """Get dividend transaction history."""
        pass
    
    # Cash Management
    @abstractmethod
    async def update_portfolio_cash(self, portfolio_id: int, new_balance: Decimal) -> bool:
        """Update portfolio cash balance."""
        pass
    
    @abstractmethod
    async def get_portfolio_cash_flow(
        self, 
        portfolio_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get cash flow history for a portfolio."""
        pass
    
    # Bulk Operations
    @abstractmethod
    async def bulk_create_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        """Create multiple transactions in bulk."""
        pass
    
    @abstractmethod
    async def bulk_update_holdings(self, holdings: List[PortfolioHolding]) -> int:
        """Update multiple holdings in bulk. Returns count of updated holdings."""
        pass
    
    # Portfolio Comparison and Benchmarking
    @abstractmethod
    async def compare_portfolios(
        self, 
        portfolio_ids: List[int],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Compare performance of multiple portfolios."""
        pass
    
    @abstractmethod
    async def get_portfolio_beta(self, portfolio_id: int, benchmark_symbol: str = "SPY") -> Optional[Decimal]:
        """Calculate portfolio beta against a benchmark."""
        pass
    
    @abstractmethod
    async def get_portfolio_sharpe_ratio(
        self, 
        portfolio_id: int,
        risk_free_rate: Decimal = Decimal('0.02')
    ) -> Optional[Decimal]:
        """Calculate portfolio Sharpe ratio."""
        pass
    
    # Risk Metrics
    @abstractmethod
    async def get_portfolio_volatility(
        self, 
        portfolio_id: int,
        days_back: int = 252
    ) -> Optional[Decimal]:
        """Calculate portfolio volatility (standard deviation of returns)."""
        pass
    
    @abstractmethod
    async def get_portfolio_var(
        self, 
        portfolio_id: int,
        confidence_level: Decimal = Decimal('0.95'),
        days_back: int = 252
    ) -> Optional[Decimal]:
        """Calculate Value at Risk for the portfolio."""
        pass
    
    @abstractmethod
    async def get_portfolio_drawdown(self, portfolio_id: int) -> Dict[str, Any]:
        """Get maximum drawdown statistics for the portfolio."""
        pass
    
    # Reporting
    @abstractmethod
    async def get_portfolio_summary(self, portfolio_id: int) -> Dict[str, Any]:
        """Get comprehensive portfolio summary for reporting."""
        pass
    
    @abstractmethod
    async def get_tax_lot_details(
        self, 
        portfolio_id: int,
        stock_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get tax lot details for holdings (FIFO/LIFO calculations)."""
        pass

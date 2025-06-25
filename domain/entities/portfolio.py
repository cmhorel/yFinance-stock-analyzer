"""Portfolio-related domain entities."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from decimal import Decimal
from enum import Enum


class TransactionType(Enum):
    """Enumeration for transaction types."""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    SPLIT = "split"


@dataclass
class Transaction:
    """Represents a portfolio transaction."""
    id: Optional[int]
    stock_id: int
    symbol: str
    transaction_type: TransactionType
    quantity: int
    price_per_share: Decimal
    total_amount: Decimal
    brokerage_fee: Decimal = Decimal('0')
    transaction_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate transaction data after initialization."""
        if self.quantity <= 0:
            raise ValueError("Transaction quantity must be positive")
        if self.price_per_share < 0:
            raise ValueError("Price per share cannot be negative")
        if self.total_amount < 0:
            raise ValueError("Total amount cannot be negative")
        if self.brokerage_fee < 0:
            raise ValueError("Brokerage fee cannot be negative")
        
        if not self.transaction_date:
            self.transaction_date = datetime.now()
        
        # Validate total amount calculation
        expected_total = (self.quantity * self.price_per_share) + self.brokerage_fee
        if abs(self.total_amount - expected_total) > Decimal('0.01'):
            raise ValueError("Total amount does not match quantity * price + fees")
    
    @property
    def net_amount(self) -> Decimal:
        """Get net amount (excluding brokerage fees)."""
        return self.quantity * self.price_per_share
    
    @property
    def is_buy(self) -> bool:
        """Check if transaction is a buy order."""
        return self.transaction_type == TransactionType.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if transaction is a sell order."""
        return self.transaction_type == TransactionType.SELL


@dataclass
class PortfolioHolding:
    """Represents a stock holding in the portfolio."""
    id: Optional[int]
    stock_id: int
    symbol: str
    quantity: int
    avg_cost_per_share: Decimal
    total_cost: Decimal
    current_price: Optional[Decimal] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate holding data after initialization."""
        if self.quantity < 0:
            raise ValueError("Holding quantity cannot be negative")
        if self.avg_cost_per_share < 0:
            raise ValueError("Average cost per share cannot be negative")
        if self.total_cost < 0:
            raise ValueError("Total cost cannot be negative")
        
        # Validate cost calculation
        if self.quantity > 0:
            expected_total = self.quantity * self.avg_cost_per_share
            if abs(self.total_cost - expected_total) > Decimal('0.01'):
                raise ValueError("Total cost does not match quantity * average cost")
    
    @property
    def current_value(self) -> Optional[Decimal]:
        """Calculate current market value of the holding."""
        if self.current_price is None:
            return None
        return self.quantity * self.current_price
    
    @property
    def unrealized_gain_loss(self) -> Optional[Decimal]:
        """Calculate unrealized gain/loss."""
        current_val = self.current_value
        if current_val is None:
            return None
        return current_val - self.total_cost
    
    @property
    def unrealized_gain_loss_percent(self) -> Optional[Decimal]:
        """Calculate unrealized gain/loss percentage."""
        gain_loss = self.unrealized_gain_loss
        if gain_loss is None or self.total_cost == 0:
            return None
        return (gain_loss / self.total_cost) * 100
    
    def update_price(self, new_price: Decimal) -> None:
        """Update the current price of the holding."""
        if new_price < 0:
            raise ValueError("Price cannot be negative")
        self.current_price = new_price
        self.last_updated = datetime.now()


@dataclass
class Portfolio:
    """Represents a complete portfolio."""
    id: Optional[int]
    name: str
    cash_balance: Decimal
    total_portfolio_value: Decimal
    last_transaction_date: Optional[datetime] = None
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    holdings: Optional[List[PortfolioHolding]] = None
    
    def __post_init__(self):
        """Initialize portfolio data after creation."""
        if not self.name or not self.name.strip():
            raise ValueError("Portfolio name cannot be empty")
        self.name = self.name.strip()
        
        if self.cash_balance < 0:
            raise ValueError("Cash balance cannot be negative")
        
        if self.holdings is None:
            self.holdings = []
        
        if not self.created_date:
            self.created_date = datetime.now()
    
    @property
    def total_holdings_value(self) -> Decimal:
        """Calculate total value of all holdings."""
        if not self.holdings:
            return Decimal('0')
        total = Decimal('0')
        for holding in self.holdings:
            current_value = holding.current_value
            if current_value is not None:
                total += current_value
        return total
    
    @property
    def total_cost_basis(self) -> Decimal:
        """Calculate total cost basis of all holdings."""
        if not self.holdings:
            return Decimal('0')
        return sum(holding.total_cost for holding in self.holdings)
    
    @property
    def total_unrealized_gain_loss(self) -> Decimal:
        """Calculate total unrealized gain/loss."""
        if not self.holdings:
            return Decimal('0')
        total = Decimal('0')
        for holding in self.holdings:
            gain_loss = holding.unrealized_gain_loss
            if gain_loss is not None:
                total += gain_loss
        return total
    
    @property
    def total_unrealized_gain_loss_percent(self) -> Optional[Decimal]:
        """Calculate total unrealized gain/loss percentage."""
        cost_basis = self.total_cost_basis
        if cost_basis == 0:
            return None
        return (self.total_unrealized_gain_loss / cost_basis) * 100
    
    @property
    def asset_allocation(self) -> dict:
        """Get asset allocation by stock symbol."""
        if not self.holdings:
            return {}
        total_value = self.total_holdings_value
        if total_value == 0:
            return {}
        
        allocation = {}
        for holding in self.holdings:
            current_value = holding.current_value
            if current_value is not None and current_value > 0:
                allocation[holding.symbol] = (current_value / total_value) * 100
        
        return allocation
    
    def add_holding(self, holding: PortfolioHolding) -> None:
        """Add a new holding to the portfolio."""
        if not self.holdings:
            self.holdings = []
        
        # Check if holding already exists
        for existing_holding in self.holdings:
            if existing_holding.stock_id == holding.stock_id:
                raise ValueError(f"Holding for stock {holding.symbol} already exists")
        
        self.holdings.append(holding)
        self.updated_date = datetime.now()
    
    def remove_holding(self, stock_id: int) -> bool:
        """Remove a holding from the portfolio."""
        if not self.holdings:
            return False
        for i, holding in enumerate(self.holdings):
            if holding.stock_id == stock_id:
                del self.holdings[i]
                self.updated_date = datetime.now()
                return True
        return False
    
    def get_holding(self, stock_id: int) -> Optional[PortfolioHolding]:
        """Get a specific holding by stock ID."""
        if not self.holdings:
            return None
        for holding in self.holdings:
            if holding.stock_id == stock_id:
                return holding
        return None
    
    def update_cash_balance(self, new_balance: Decimal) -> None:
        """Update the cash balance."""
        if new_balance < 0:
            raise ValueError("Cash balance cannot be negative")
        self.cash_balance = new_balance
        self.updated_date = datetime.now()

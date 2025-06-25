"""SQLite implementation of the portfolio repository."""
import sqlite3
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from contextlib import asynccontextmanager

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from domain.repositories.portfolio_repository import IPortfolioRepository
from domain.entities.portfolio import Portfolio, PortfolioHolding, Transaction, TransactionType
from shared.config import get_settings
from shared.logging import get_logger
from shared.exceptions.data import DataNotFoundError, DataValidationError


class SqlitePortfolioRepository(IPortfolioRepository):
    """SQLite implementation of portfolio repository."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.settings = get_settings()
        self.db_path = db_path or self.settings.database.path
        self.logger = get_logger(__name__)
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create portfolio tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cash_balance DECIMAL(15,2) NOT NULL,
                    total_portfolio_value DECIMAL(15,2) NOT NULL,
                    last_transaction_date DATE,
                    created_date DATE UNIQUE NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS portfolio_holdings (
                    stock_id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost_per_share DECIMAL(10,4) NOT NULL,
                    total_cost DECIMAL(15,2) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (stock_id) REFERENCES stocks(id)
                );
                
                CREATE TABLE IF NOT EXISTS portfolio_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('BUY', 'SELL')),
                    quantity INTEGER NOT NULL,
                    price_per_share DECIMAL(10,4) NOT NULL,
                    total_amount DECIMAL(15,2) NOT NULL,
                    brokerage_fee DECIMAL(10,2) DEFAULT 0.00,
                    transaction_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (stock_id) REFERENCES stocks(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_portfolio_transactions_date ON portfolio_transactions(transaction_date);
                CREATE INDEX IF NOT EXISTS idx_portfolio_transactions_stock ON portfolio_transactions(stock_id);
                CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_symbol ON portfolio_holdings(symbol);
            """)
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _row_to_portfolio(self, row: sqlite3.Row) -> Portfolio:
        """Convert database row to Portfolio entity."""
        return Portfolio(
            id=row['id'],
            name="Default Portfolio",
            cash_balance=Decimal(str(row['cash_balance'])),
            total_portfolio_value=Decimal(str(row['total_portfolio_value'])),
            last_transaction_date=datetime.fromisoformat(row['last_transaction_date']) if row['last_transaction_date'] else None,
            created_date=datetime.fromisoformat(row['created_date']) if row['created_date'] else None,
            updated_date=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        )
    
    def _row_to_holding(self, row: sqlite3.Row) -> PortfolioHolding:
        """Convert database row to PortfolioHolding entity."""
        return PortfolioHolding(
            id=None,
            stock_id=row['stock_id'],
            symbol=row['symbol'],
            quantity=row['quantity'],
            avg_cost_per_share=Decimal(str(row['avg_cost_per_share'])),
            total_cost=Decimal(str(row['total_cost'])),
            last_updated=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        )
    
    def _row_to_transaction(self, row: sqlite3.Row) -> Transaction:
        """Convert database row to Transaction entity."""
        return Transaction(
            id=row['id'],
            stock_id=row['stock_id'],
            symbol=row['symbol'],
            transaction_type=TransactionType(row['transaction_type'].upper()),
            quantity=row['quantity'],
            price_per_share=Decimal(str(row['price_per_share'])),
            total_amount=Decimal(str(row['total_amount'])),
            brokerage_fee=Decimal(str(row['brokerage_fee'])) if row['brokerage_fee'] else Decimal('0'),
            transaction_date=datetime.fromisoformat(row['transaction_date']) if row['transaction_date'] else None,
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
        )
    
    # Portfolio Operations
    async def get_current_portfolio(self) -> Optional[Portfolio]:
        """Get the current portfolio state."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_state ORDER BY created_date DESC, id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return self._row_to_portfolio(row) if row else None
    
    async def create_portfolio_state(self, portfolio: Portfolio) -> Portfolio:
        """Create a new portfolio state record."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO portfolio_state 
                   (cash_balance, total_portfolio_value, last_transaction_date, created_date, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (float(portfolio.cash_balance), float(portfolio.total_portfolio_value),
                 portfolio.last_transaction_date.isoformat() if portfolio.last_transaction_date else None,
                 portfolio.created_date.isoformat() if portfolio.created_date else datetime.now().isoformat(), 
                 datetime.now().isoformat())
            )
            conn.commit()
            
            portfolio.id = cursor.lastrowid
            portfolio.updated_date = datetime.now()
            
            self.logger.info(f"Created portfolio state: ID {portfolio.id}")
            return portfolio
    
    async def update_portfolio_state(self, portfolio: Portfolio) -> Portfolio:
        """Update an existing portfolio state."""
        async with self._get_connection() as conn:
            conn.execute(
                """UPDATE portfolio_state SET 
                   cash_balance = ?, total_portfolio_value = ?, 
                   last_transaction_date = ?, updated_at = ?
                   WHERE id = ?""",
                (float(portfolio.cash_balance), float(portfolio.total_value),
                 portfolio.last_transaction_date.isoformat() if portfolio.last_transaction_date else None,
                 datetime.now().isoformat(), portfolio.id)
            )
            conn.commit()
            
            portfolio.updated_at = datetime.now()
            self.logger.info(f"Updated portfolio state: ID {portfolio.id}")
            return portfolio
    
    async def get_portfolio_history(
        self, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[Portfolio]:
        """Get portfolio history within date range."""
        query = "SELECT * FROM portfolio_state WHERE 1=1"
        params: List[Any] = []
        
        if start_date:
            query += " AND created_date >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND created_date <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY created_date ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        async with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_portfolio(row) for row in cursor.fetchall()]
    
    # Holdings Operations
    async def get_current_holdings(self) -> List[PortfolioHolding]:
        """Get all current portfolio holdings."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_holdings WHERE quantity > 0 ORDER BY symbol"
            )
            return [self._row_to_holding(row) for row in cursor.fetchall()]
    
    async def get_holding_by_stock_id(self, stock_id: int) -> Optional[PortfolioHolding]:
        """Get a specific holding by stock ID."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_holdings WHERE stock_id = ?", (stock_id,)
            )
            row = cursor.fetchone()
            return self._row_to_holding(row) if row else None
    
    async def get_holding_by_symbol(self, symbol: str) -> Optional[PortfolioHolding]:
        """Get a specific holding by symbol."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_holdings WHERE symbol = ?", (symbol.upper(),)
            )
            row = cursor.fetchone()
            return self._row_to_holding(row) if row else None
    
    async def create_or_update_holding(self, holding: PortfolioHolding) -> PortfolioHolding:
        """Create or update a portfolio holding."""
        async with self._get_connection() as conn:
            # Try to update first
            cursor = conn.execute(
                """UPDATE portfolio_holdings SET 
                   quantity = ?, avg_cost_per_share = ?, total_cost = ?, updated_at = ?
                   WHERE stock_id = ?""",
                (holding.quantity, float(holding.average_cost), float(holding.total_cost),
                 datetime.now().isoformat(), holding.stock_id)
            )
            
            if cursor.rowcount == 0:
                # Insert new record
                conn.execute(
                    """INSERT INTO portfolio_holdings 
                       (stock_id, symbol, quantity, avg_cost_per_share, total_cost, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (holding.stock_id, holding.symbol.upper(), holding.quantity,
                     float(holding.average_cost), float(holding.total_cost),
                     datetime.now().isoformat())
                )
            
            conn.commit()
            holding.updated_at = datetime.now()
            
            self.logger.info(f"Updated holding: {holding.symbol} ({holding.quantity} shares)")
            return holding
    
    async def delete_holding(self, stock_id: int) -> bool:
        """Delete a portfolio holding."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM portfolio_holdings WHERE stock_id = ?", (stock_id,)
            )
            conn.commit()
            
            deleted = cursor.rowcount > 0
            if deleted:
                self.logger.info(f"Deleted holding for stock ID: {stock_id}")
            return deleted
    
    # Transaction Operations
    async def create_transaction(self, transaction: Transaction) -> Transaction:
        """Create a new transaction record."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO portfolio_transactions 
                   (stock_id, symbol, transaction_type, quantity, price_per_share, 
                    total_amount, brokerage_fee, transaction_date, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (transaction.stock_id, transaction.symbol.upper(), transaction.transaction_type.value,
                 transaction.quantity, float(transaction.price_per_share), float(transaction.total_amount),
                 float(transaction.brokerage_fee), transaction.transaction_date.isoformat(),
                 datetime.now().isoformat())
            )
            conn.commit()
            
            transaction.id = cursor.lastrowid
            transaction.created_at = datetime.now()
            
            self.logger.info(f"Created transaction: {transaction.transaction_type.value} {transaction.quantity} {transaction.symbol}")
            return transaction
    
    async def get_transactions(
        self,
        stock_id: Optional[int] = None,
        symbol: Optional[str] = None,
        transaction_type: Optional[TransactionType] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[Transaction]:
        """Get transactions with optional filters."""
        query = "SELECT * FROM portfolio_transactions WHERE 1=1"
        params: List[Any] = []
        
        if stock_id:
            query += " AND stock_id = ?"
            params.append(stock_id)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())
        
        if transaction_type:
            query += " AND transaction_type = ?"
            params.append(transaction_type.value)
        
        if start_date:
            query += " AND transaction_date >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND transaction_date <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY transaction_date DESC, created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        async with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_transaction(row) for row in cursor.fetchall()]
    
    async def get_transaction_by_id(self, transaction_id: int) -> Optional[Transaction]:
        """Get a specific transaction by ID."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_transactions WHERE id = ?", (transaction_id,)
            )
            row = cursor.fetchone()
            return self._row_to_transaction(row) if row else None
    
    # Analytics Operations
    async def get_portfolio_performance(
        self, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        async with self._get_connection() as conn:
            # Get portfolio states in date range
            query = "SELECT * FROM portfolio_state WHERE 1=1"
            params: List[Any] = []
            
            if start_date:
                query += " AND created_date >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND created_date <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY created_date ASC"
            
            cursor = conn.execute(query, params)
            states = [self._row_to_portfolio(row) for row in cursor.fetchall()]
            
            if not states:
                return {'error': 'No portfolio data found'}
            
            initial_value = states[0].total_value
            current_value = states[-1].total_value
            total_return = current_value - initial_value
            return_percentage = (total_return / initial_value * 100) if initial_value > 0 else 0
            
            return {
                'initial_value': initial_value,
                'current_value': current_value,
                'total_return': total_return,
                'return_percentage': return_percentage,
                'data_points': len(states),
                'start_date': states[0].created_date,
                'end_date': states[-1].created_date
            }
    
    async def get_holding_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance metrics for a specific holding."""
        async with self._get_connection() as conn:
            # Get all transactions for this symbol
            cursor = conn.execute(
                """SELECT * FROM portfolio_transactions 
                   WHERE symbol = ? ORDER BY transaction_date ASC""",
                (symbol.upper(),)
            )
            transactions = [self._row_to_transaction(row) for row in cursor.fetchall()]
            
            if not transactions:
                return {'error': f'No transactions found for {symbol}'}
            
            # Calculate metrics
            total_bought = sum(t.quantity for t in transactions if t.transaction_type == TransactionType.BUY)
            total_sold = sum(t.quantity for t in transactions if t.transaction_type == TransactionType.SELL)
            current_quantity = total_bought - total_sold
            
            total_cost = sum(t.total_amount for t in transactions if t.transaction_type == TransactionType.BUY)
            total_proceeds = sum(t.total_amount for t in transactions if t.transaction_type == TransactionType.SELL)
            
            avg_buy_price = total_cost / total_bought if total_bought > 0 else Decimal('0')
            avg_sell_price = total_proceeds / total_sold if total_sold > 0 else Decimal('0')
            
            return {
                'symbol': symbol,
                'total_bought': total_bought,
                'total_sold': total_sold,
                'current_quantity': current_quantity,
                'total_cost': total_cost,
                'total_proceeds': total_proceeds,
                'avg_buy_price': avg_buy_price,
                'avg_sell_price': avg_sell_price,
                'realized_gain_loss': total_proceeds - (total_sold * avg_buy_price) if total_sold > 0 else Decimal('0'),
                'transaction_count': len(transactions),
                'first_transaction': transactions[0].transaction_date,
                'last_transaction': transactions[-1].transaction_date
            }
    
    async def get_transaction_summary(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get transaction summary statistics."""
        async with self._get_connection() as conn:
            query = """SELECT 
                       transaction_type,
                       COUNT(*) as count,
                       SUM(quantity) as total_quantity,
                       SUM(total_amount) as total_amount,
                       AVG(price_per_share) as avg_price,
                       SUM(brokerage_fee) as total_fees
                       FROM portfolio_transactions WHERE 1=1"""
            params: List[Any] = []
            
            if start_date:
                query += " AND transaction_date >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND transaction_date <= ?"
                params.append(end_date.isoformat())
            
            query += " GROUP BY transaction_type"
            
            cursor = conn.execute(query, params)
            results = {}
            
            for row in cursor.fetchall():
                transaction_type = row['transaction_type']
                results[transaction_type.lower()] = {
                    'count': row['count'],
                    'total_quantity': row['total_quantity'],
                    'total_amount': Decimal(str(row['total_amount'])),
                    'avg_price': Decimal(str(row['avg_price'])),
                    'total_fees': Decimal(str(row['total_fees']))
                }
            
            return results
    
    # Utility Operations
    async def initialize_portfolio(self, initial_cash: Decimal = Decimal('10000')) -> Portfolio:
        """Initialize a new portfolio with starting cash."""
        today = date.today()
        
        # Check if portfolio already exists for today
        existing = await self.get_current_portfolio()
        if existing and existing.created_date == today:
            return existing
        
        portfolio = Portfolio(
            id=None,
            cash_balance=initial_cash,
            total_value=initial_cash,
            last_transaction_date=None,
            created_date=today,
            updated_at=None
        )
        
        return await self.create_portfolio_state(portfolio)
    
    async def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """Clean up old portfolio data beyond specified days."""
        cutoff_date = date.today() - timedelta(days=days_to_keep)
        
        async with self._get_connection() as conn:
            # Delete old portfolio states (keep at least one)
            cursor = conn.execute(
                """DELETE FROM portfolio_state 
                   WHERE created_date < ? AND id NOT IN (
                       SELECT id FROM portfolio_state ORDER BY created_date DESC LIMIT 1
                   )""",
                (cutoff_date.isoformat(),)
            )
            deleted_states = cursor.rowcount
            
            # Delete old transactions
            cursor = conn.execute(
                "DELETE FROM portfolio_transactions WHERE transaction_date < ?",
                (cutoff_date.isoformat(),)
            )
            deleted_transactions = cursor.rowcount
            
            conn.commit()
            
            total_deleted = deleted_states + deleted_transactions
            if total_deleted > 0:
                self.logger.info(f"Cleaned up {total_deleted} old records (states: {deleted_states}, transactions: {deleted_transactions})")
            
            return total_deleted

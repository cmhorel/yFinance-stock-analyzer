# database/schema.py
from database.connection import db_manager

SCHEMA_SQL = """
-- Stocks table
CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    name TEXT,
    exchange TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stock prices table with improved indexing
CREATE TABLE IF NOT EXISTS stock_prices (
    stock_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    adjusted_close REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, date),
    FOREIGN KEY (stock_id) REFERENCES stocks(id) ON DELETE CASCADE
);

-- Stock information table
CREATE TABLE IF NOT EXISTS stock_info (
    stock_id INTEGER PRIMARY KEY,
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    beta REAL,
    pe_ratio REAL,
    dividend_yield REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id) ON DELETE CASCADE
);

-- News and sentiment table
CREATE TABLE IF NOT EXISTS stock_news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    sentiment_score REAL,
    confidence_score REAL,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (stock_id, date, title),
    FOREIGN KEY (stock_id) REFERENCES stocks(id) ON DELETE CASCADE
);

-- Portfolio state table
CREATE TABLE IF NOT EXISTS portfolio_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cash_balance REAL NOT NULL,
    total_portfolio_value REAL NOT NULL,
    last_transaction_date TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio holdings table
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    stock_id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_cost_per_share REAL NOT NULL,
    total_cost REAL NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id) ON DELETE CASCADE
);

-- Portfolio transactions table
CREATE TABLE IF NOT EXISTS portfolio_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL,
    price_per_share REAL NOT NULL,
    total_amount REAL NOT NULL,
    brokerage_fee REAL NOT NULL,
    transaction_date TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks(id) ON DELETE CASCADE
);

-- Technical indicators cache table
CREATE TABLE IF NOT EXISTS technical_indicators (
    stock_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    rsi_14 REAL,
    ma_20 REAL,
    ma_50 REAL,
    ma_200 REAL,
    bollinger_upper REAL,
    bollinger_lower REAL,
    macd REAL,
    macd_signal REAL,
    volume_sma_20 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, date),
    FOREIGN KEY (stock_id) REFERENCES stocks(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(date);
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(stock_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_news_date ON stock_news(stock_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_transactions_date ON portfolio_transactions(transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_date ON technical_indicators(stock_id, date DESC);
"""

def initialize_database():
    """Initialize the database with the complete schema."""
    db_manager.execute_script(SCHEMA_SQL)
    
    # Initialize portfolio if it doesn't exist
    existing_portfolio = db_manager.execute_query(
        "SELECT COUNT(*) as count FROM portfolio_state"
    )
    
    if existing_portfolio[0]['count'] == 0:
        from config import config
        db_manager.execute_query(
            """INSERT INTO portfolio_state (cash_balance, total_portfolio_value, last_transaction_date)
               VALUES (?, ?, NULL)""",
            (config.trading.initial_capital, config.trading.initial_capital)
        )
        print(f"Initialized portfolio with ${config.trading.initial_capital:,.2f}")
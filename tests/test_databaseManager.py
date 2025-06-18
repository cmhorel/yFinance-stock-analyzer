import unittest, pytest
from datetime import datetime, timedelta
import pandas as pd
from app.database_manager import DatabaseManager

@pytest.fixture(scope="module")
def db():
    db = DatabaseManager("file::memory:?cache=shared")
    with db.get_connection() as conn:
        # ... create tables as before ...
        conn.executescript("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS stock_info (
                    stock_id INTEGER PRIMARY KEY,
                    sector TEXT,
                    industry TEXT
                );
                CREATE TABLE IF NOT EXISTS stock_prices (
                    stock_id INTEGER,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER
                );
                CREATE TABLE IF NOT EXISTS stock_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_id INTEGER,
                    date TEXT,
                    title TEXT,
                    summary TEXT,
                    sentiment_score REAL
                );
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cash_balance REAL,
                    total_portfolio_value REAL,
                    last_transaction_date TEXT,
                    created_date TEXT
                );
                CREATE TABLE IF NOT EXISTS portfolio_holdings (
                    stock_id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    quantity INTEGER,
                    avg_cost_per_share REAL,
                    total_cost REAL
                );
                CREATE TABLE IF NOT EXISTS portfolio_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_id INTEGER,
                    symbol TEXT,
                    transaction_type TEXT,
                    quantity INTEGER,
                    price_per_share REAL,
                    total_amount REAL,
                    brokerage_fee REAL,
                    transaction_date TEXT
                );
            """)
        conn.commit()
    yield db
    # No teardown needed for in-memory DB

@pytest.fixture(autouse=True)
def truncate_tables(db):
    with db.get_connection() as conn:
        cursor = conn.cursor()
        # List all your tables here
        tables = [
            "stocks",
            "stock_info",
            "stock_prices",
            "stock_news",
            "portfolio_state",
            "portfolio_holdings",
            "portfolio_transactions"
        ]
        for table in tables:
            cursor.execute(f"DELETE FROM {table};")
        conn.commit()
    yield

def test_get_or_create_stock_id( db):
    stock_id1 = db.get_or_create_stock_id("AAPL")
    stock_id2 = db.get_or_create_stock_id("AAPL")
    assert stock_id1 == stock_id2
    stock_id3 = db.get_or_create_stock_id("MSFT")
    assert stock_id1 != stock_id3

def test_store_and_get_stock_info(db):
    stock_id = db.get_or_create_stock_id("AAPL")
    db.store_stock_info(stock_id, "Tech", "Software")
    ids = db.get_industry_stocks("Software")
    assert stock_id in ids
    ids_exclude = db.get_industry_stocks("Software", exclude_stock_id=stock_id)
    assert stock_id not in ids_exclude

def test_store_and_get_stock_prices(db):
    stock_id = db.get_or_create_stock_id("AAPL")
    today = datetime.now().strftime('%Y-%m-%d')
    price_data = [{
        'date': today, 'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 1000
    }]
    db.store_stock_prices(stock_id, price_data)
    latest = db.get_latest_stock_price_date("AAPL")
    assert latest == today

def test_store_news_sentiments_and_average(db):
    stock_id = db.get_or_create_stock_id("AAPL")
    today = datetime.now().strftime('%Y-%m-%d')
    news = [
        {'publish_date': today, 'title': 'T1', 'summary': 'S1', 'sentiment_score': 0.5},
        {'publish_date': today, 'title': 'T2', 'summary': 'S2', 'sentiment_score': 1.0},
    ]
    db.store_news_sentiments(stock_id, news)
    avg = db.get_average_sentiment(stock_id, days_back=1)
    assert abs(avg - 0.75) < 1e-2

def test_get_sentiment_timeseries(db):
    stock_id = db.get_or_create_stock_id("AAPL")
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    news = [
        {'publish_date': yesterday, 'title': 'T1', 'summary': 'S1', 'sentiment_score': 0.5},
        {'publish_date': today, 'title': 'T2', 'summary': 'S2', 'sentiment_score': 1.0},
    ]
    db.store_news_sentiments(stock_id, news)
    df = db.get_sentiment_timeseries(stock_id, days_back=2)
    assert len(df) == 2
    assert 'avg_sentiment' in df.columns
    assert (df['avg_sentiment'] > 0).any()

def test_update_or_create_holding(db):
    stock_id = db.get_or_create_stock_id("AAPL")
    db.update_or_create_holding(stock_id, "AAPL", 10, 100.0, 1000.0)
    db.update_or_create_holding(stock_id, "AAPL", 5, 110.0, 550.0)
    with db.get_connection() as conn:
        cur = conn.execute('SELECT quantity, avg_cost_per_share, total_cost FROM portfolio_holdings WHERE stock_id=?', (stock_id,))
        row = cur.fetchone()
        assert row[0] == 15
        assert abs(row[1] - (1000.0 + 550.0) / 15) < 1e-2
        assert abs(row[2] - 1550.0) < 1e-2

def test_reduce_or_remove_holding(db):
    stock_id = db.get_or_create_stock_id("AAPL")
    db.update_or_create_holding(stock_id, "AAPL", 10, 100.0, 1000.0)
    db.reduce_or_remove_holding(stock_id, 5)
    with db.get_connection() as conn:
        cur = conn.execute('SELECT quantity FROM portfolio_holdings WHERE stock_id=?', (stock_id,))
        row = cur.fetchone()
        assert row[0] == 5
    db.reduce_or_remove_holding(stock_id, 5)
    with db.get_connection() as conn:
        cur = conn.execute('SELECT COUNT(*) FROM portfolio_holdings WHERE stock_id=?', (stock_id,))
        count = cur.fetchone()[0]
        assert count == 0

def test_record_transaction_and_get_transactions_df(db):
    stock_id = db.get_or_create_stock_id("AAPL")
    today = datetime.now().strftime('%Y-%m-%d')
    db.record_transaction(stock_id, "AAPL", "BUY", 10, 100.0, 1000.0, 1.0, today)
    df = db.get_transactions_df()
    assert len(df) == 1
    assert df.iloc[0]['transaction_type'] == "BUY"

def test_portfolio_state_and_cash_update(db):
    # Insert a portfolio state row
    with db.get_connection() as conn:
        conn.execute('''
            INSERT INTO portfolio_state (cash_balance, total_portfolio_value, last_transaction_date, created_date)
            VALUES (?, ?, ?, ?)
        ''', (10000.0, 10000.0, "2024-01-01", "2024-01-01"))
        conn.commit()
    db.update_portfolio_cash(9000.0, "2024-01-02")
    with db.get_connection() as conn:
        cur = conn.execute('SELECT cash_balance, last_transaction_date FROM portfolio_state ORDER BY id DESC LIMIT 1')
        row = cur.fetchone()
        assert row[0] == 9000.0
        assert row[1] == "2024-01-02"
    db.update_portfolio_value(9500.0)
    with db.get_connection() as conn:
        cur = conn.execute('SELECT total_portfolio_value FROM portfolio_state ORDER BY id DESC LIMIT 1')
        row = cur.fetchone()
        assert row[0] == 9500.0

if __name__ == "__main__":
    unittest.main()
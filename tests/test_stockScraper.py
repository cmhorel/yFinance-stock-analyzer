import pytest
from unittest.mock import patch, MagicMock
import app.stockScraper as stockScraper
import unittest
import os
import sqlite3
from datetime import datetime
import pandas as pd

class DummyDBManager:
    def __init__(self):
        self.stored = None

    def get_or_create_stock_id(self, ticker):
        return 1

    def get_latest_date(self, ticker):
        return None

    def get_latest_stock_close_price(self, ticker):
        return 10  # or whatever makes sense for your test

    def store_stock_prices(self, stock_id, price_data):
        self.stored = (stock_id, price_data)

    def store_stock_info(self, stock_id, sector, industry):
        pass

    def get_latest_stock_price_date(self, symbol):
        return "2023-12-31"

@pytest.fixture
def mock_db_manager(monkeypatch):
    dummy = DummyDBManager()
    monkeypatch.setattr(stockScraper, "db_manager", dummy)
    return dummy

@patch("app.stockScraper.yf.download")
def test_sync_ticker_inserts_prices(mock_yf_download, mock_db_manager):
    # Prepare fake price data
    import pandas as pd
    idx = pd.date_range("2024-01-01", periods=2)
    df = pd.DataFrame({
        "Open": [10, 11],
        "High": [12, 13],
        "Low": [9, 10],
        "Close": [11, 12],
        "Volume": [1000, 1100]
    }, index=idx)
    mock_yf_download.return_value = df

    with patch("app.stockScraper.db_manager", new=DummyDBManager()), \
         patch("app.stockScraper.news_analyzer.get_industry_and_sector", return_value={"sector": "Tech", "industry": "Software"}), \
         patch("app.stockScraper.news_analyzer.process_stock_news"):
        stockScraper.sync_ticker("AAPL")
        assert hasattr(stockScraper.db_manager, "stored")
        stock_id, price_data = stockScraper.db_manager.stored
        print(stock_id, price_data)
        assert stock_id == 1
        assert len(price_data) == 2
        assert price_data[0]["open"] == 10

@patch("app.stockScraper.yf.download")
def test_sync_ticker_handles_no_data(mock_yf_download, mock_db_manager):
    mock_yf_download.return_value = MagicMock(empty=True)
    with patch("app.stockScraper.news_analyzer.get_industry_and_sector", return_value=None), \
         patch("app.stockScraper.news_analyzer.process_stock_news"):
        # Should not raise
        stockScraper.sync_ticker("FAKE")

def test_get_nasdaq_100_tickers(monkeypatch):
    # Patch pd.read_html to return a fake table
    import pandas as pd
    monkeypatch.setattr(pd, "read_html", lambda url: [None, None, None, None, pd.DataFrame({"Ticker": ["AAPL", "MSFT"]})])
    tickers = stockScraper.get_nasdaq_100_tickers()
    assert tickers == ["AAPL", "MSFT"]

def test_get_tsx60_tickers(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(pd, "read_html", lambda url: [pd.DataFrame({"Symbol": ["BNS", "RY"]})])
    tickers = stockScraper.get_tsx60_tickers()
    assert tickers == ["BNS.TO", "RY.TO"]

def test_get_tsx60_tickers_raises(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(pd, "read_html", lambda url: [pd.DataFrame({"NotSymbol": ["BNS", "RY"]})])
    with pytest.raises(ValueError):
        stockScraper.get_tsx60_tickers()

def test_get_latest_date(monkeypatch, mock_db_manager):
    monkeypatch.setattr(stockScraper.db_manager, "get_latest_stock_price_date", lambda symbol: "2024-01-01")
    assert stockScraper.get_latest_date("AAPL") == "2024-01-01"

def test_plot_all(monkeypatch):
    # Patch out plt.show and DB access
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    class DummyCursor:
        def execute(self, *a, **k): return self
        def fetchall(self): return [("2024-01-01", 100.0), ("2024-01-02", 101.0)]
    class DummyConn:
        def cursor(self): return DummyCursor()
        def close(self): pass
    monkeypatch.setattr(stockScraper.sqlite3, "connect", lambda db: DummyConn())
    stockScraper.plot_all(["AAPL"])

def test_initialize_db(monkeypatch):
    # Patch out sqlite3.connect
    class DummyConn:
        def cursor(self): return self
        def execute(self, *a, **k): return None
        def commit(self): return None
        def close(self): return None
    monkeypatch.setattr(stockScraper.sqlite3, "connect", lambda db: DummyConn())
    stockScraper.initialize_db()




# Python

from app.stockScraper import (
    initialize_db, get_latest_date, sync_batch, get_nasdaq_100_tickers,
    get_tsx60_tickers, plot_all, DB_NAME
)

class TestStockScraper(unittest.TestCase):
    TEST_DB = "test_stockScraper.db"

    def setUp(self):
        self.TEST_DB = "test_stockScraper.db"
        # Patch DB_NAME to use test DB
        self.db_name_patch = patch("app.stockScraper.DB_NAME", self.TEST_DB)
        self.db_name_patch.start()
        # Patch db_manager to use test DB
        from app.stockScraper import DatabaseManager
        self.db_manager_patch = patch("app.stockScraper.db_manager", DatabaseManager(self.TEST_DB))
        self.db_manager_patch.start()
        # Remove test DB if exists
        if os.path.exists(self.TEST_DB):
            os.remove(self.TEST_DB)
        initialize_db()

    def tearDown(self):
        self.db_name_patch.stop()
        self.db_manager_patch.stop()
        # Explicitly close the db_manager connection if possible
        try:
            from app.stockScraper import db_manager
            db_manager.close()
        except Exception:
            pass
        import gc
        gc.collect()
        if os.path.exists(self.TEST_DB):
            try:
                os.remove(self.TEST_DB)
            except PermissionError:
                import time
                time.sleep(0.2)
                os.remove(self.TEST_DB)

    def test_initialize_db_creates_tables(self):
        conn = sqlite3.connect(self.TEST_DB)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in c.fetchall()}
        expected = {'stocks', 'stock_prices', 'stock_info', 'stock_news'}
        self.assertTrue(expected.issubset(tables))
        conn.close()

    def test_get_latest_date_none_and_after_insert(self):
        symbol = "FAKE"
        # Should be None before any insert
        self.assertIsNone(get_latest_date(symbol))
        # Insert a price row
        conn = sqlite3.connect(self.TEST_DB)
        c = conn.cursor()
        c.execute("INSERT INTO stocks(symbol) VALUES (?)", (symbol,))
        stock_id = c.execute("SELECT id FROM stocks WHERE symbol=?", (symbol,)).fetchone()[0]
        c.execute("INSERT INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (stock_id, "2024-01-01", 1, 2, 3, 4, 5))
        conn.commit()
        conn.close()
        self.assertEqual(get_latest_date(symbol), "2024-01-01")

    @patch("app.stockScraper.yf_download_with_retry")
    def test_sync_batch_inserts_prices(self, mock_yf):
        # Mock yfinance DataFrame
        idx = pd.date_range("2024-01-01", periods=2)
        df = pd.DataFrame({
            ("FAKE", "Open"): [1, 2],
            ("FAKE", "High"): [2, 3],
            ("FAKE", "Low"): [0.5, 1.5],
            ("FAKE", "Close"): [1.5, 1.6],
            ("FAKE", "Volume"): [100, 200],
        }, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        mock_yf.return_value = df
        sync_batch(["FAKE"])
        # Check DB for inserted prices
        conn = sqlite3.connect(self.TEST_DB)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM stock_prices")
        count = c.fetchone()[0]
        self.assertEqual(count, 2)
        conn.close()

    @patch("app.stockScraper.pd.read_html")
    def test_get_nasdaq_100_tickers(self, mock_html):
        # Mock Wikipedia table
        mock_html.return_value = [None, None, None, None, pd.DataFrame({"Ticker": ["AAPL", "MSFT"]})]
        tickers = get_nasdaq_100_tickers()
        self.assertEqual(tickers, ["AAPL", "MSFT"])

    @patch("app.stockScraper.pd.read_html")
    def test_get_tsx60_tickers(self, mock_html):
        df = pd.DataFrame({"Symbol": ["BNS", "RY"]})
        mock_html.return_value = [df]
        tickers = get_tsx60_tickers()
        self.assertEqual(tickers, ["BNS.TO", "RY.TO"])

    @patch("app.stockScraper.plt.show")
    def test_plot_all_runs(self, mock_show):
        # Insert fake stock and price
        conn = sqlite3.connect(self.TEST_DB)
        c = conn.cursor()
        c.execute("INSERT INTO stocks(symbol) VALUES ('FAKE')")
        stock_id = c.execute("SELECT id FROM stocks WHERE symbol='FAKE'").fetchone()[0]
        c.execute("INSERT INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (stock_id, "2024-01-01", 1, 2, 0.5, 1.5, 100))
        conn.commit()
        conn.close()
        # Should not raise
        plot_all(["FAKE"])
        mock_show.assert_called_once()

    @patch("app.stockScraper.yf_download_with_retry")
    def test_sync_batch_rejects_large_price_jump(self, mock_yf):
        """
        If the close price is >10% different from the previous day and it gets stored, fail the test.
        """
        # Insert initial price row for FAKE
        conn = sqlite3.connect(self.TEST_DB)
        c = conn.cursor()
        c.execute("INSERT INTO stocks(symbol) VALUES ('FAKE')")
        stock_id = c.execute("SELECT id FROM stocks WHERE symbol='FAKE'").fetchone()[0]
        c.execute("INSERT INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (stock_id, "2024-01-01", 10.0, 12.0, 9.0, 10.0, 100))
        conn.commit()
        conn.close()

        # Mock yfinance DataFrame with a >10% jump in close price
        idx = pd.date_range("2024-01-02", periods=1)
        df = pd.DataFrame({
            ("FAKE", "Open"): [11.0],
            ("FAKE", "High"): [13.0],
            ("FAKE", "Low"): [10.0],
            ("FAKE", "Close"): [12.0],  # 20% jump from previous close of 10
            ("FAKE", "Volume"): [120],
        }, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        mock_yf.return_value = df

        # Run sync_batch
        sync_batch(["FAKE"])

        # Check if the new price was stored
        conn = sqlite3.connect(self.TEST_DB)
        c = conn.cursor()
        c.execute("SELECT close FROM stock_prices WHERE stock_id=? ORDER BY date", (stock_id,))
        closes = [row[0] for row in c.fetchall()]
        conn.close()

        # There should only be the original close price if the jump was rejected
        # If the new close price (12) is present, fail the test\
        print(closes)
        self.assertNotIn(12, closes, "Close price with >10% jump was incorrectly accepted.")

    def test_sync_ticker_inserts_prices(self):
        mock_db_manager = MagicMock()
        mock_db_manager.get_latest_stock_price_date.return_value = None  
        with patch("app.stockScraper.db_manager", mock_db_manager), \
             patch("app.stockScraper.yf_download_with_retry") as mock_yf, \
             patch("app.stockScraper.validate_price_data") as mock_validate:
            df = pd.DataFrame({
                'Open': [100],
                'High': [110],
                'Low': [90],
                'Close': [105],
                'Volume': [1000]
            }, index=[pd.Timestamp("2023-01-01")])
            mock_yf.return_value = df
            mock_validate.return_value = df
            stockScraper.sync_ticker("AAPL")
            self.assertTrue(mock_db_manager.store_stock_prices.called)

if __name__ == "__main__":
    unittest.main()
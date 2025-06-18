import pytest
from unittest.mock import patch, MagicMock
from app import stockScraper

@pytest.fixture
def mock_db_manager(monkeypatch):
    class DummyDBManager:
        def get_or_create_stock_id(self, symbol):
            return 1
        def get_latest_stock_price_date(self, symbol):
            return None
        def store_stock_prices(self, stock_id, price_data):
            self.stored = (stock_id, price_data)
        def store_stock_info(self, stock_id, sector, industry):
            self.info = (stock_id, sector, industry)
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

    with patch("app.stockScraper.news_analyzer.get_industry_and_sector", return_value={"sector": "Tech", "industry": "Software"}), \
         patch("app.stockScraper.news_analyzer.process_stock_news"):
        stockScraper.sync_ticker("AAPL")
        # Check that store_stock_prices was called with expected data
        assert hasattr(stockScraper.db_manager, "stored")
        stock_id, price_data = stockScraper.db_manager.stored
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
import pytest, datetime
from unittest.mock import patch, MagicMock
from app.news_analyzer.data_fetcher import DataFetcher

def test_fetch_recent_news_returns_news_list():
    # Patch yf.Ticker to return a mock with a .news attribute
    with patch("app.news_analyzer.data_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = [
            {"content": {"pubDate": datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                          "title": "Headline",
                            "summary": "Summary"}}
        ]
        news = DataFetcher.fetch_recent_news("AAPL", days_back=7)
        assert isinstance(news, list)
        assert len(news) == 1
        assert "content" in news[0]
        assert "pubDate" in news[0]["content"]

def test_fetch_recent_news_handles_no_news():
    with patch("app.news_analyzer.data_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = []
        news = DataFetcher.fetch_recent_news("AAPL", days_back=7)
        assert isinstance(news, list)
        assert len(news) == 0

def test_fetch_recent_news_handles_exception():
    with patch("app.news_analyzer.data_fetcher.yf.Ticker", side_effect=Exception("API error")):
        news = DataFetcher.fetch_recent_news("AAPL", days_back=7)
        assert news == []

def test_get_stock_info_returns_sector_and_industry():
    with patch("app.news_analyzer.data_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {"sector": "Tech", "industry": "Software"}
        info = DataFetcher.get_stock_info("AAPL")
        assert isinstance(info, dict)
        assert info["sector"] == "Tech"
        assert info["industry"] == "Software"

def test_get_stock_info_handles_missing_keys():
    with patch("app.news_analyzer.data_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {}
        info = DataFetcher.get_stock_info("AAPL")
        assert info["sector"] == "Unknown"
        assert info["industry"] == "Unknown"

def test_get_stock_info_handles_exception():
    with patch("app.news_analyzer.data_fetcher.yf.Ticker", side_effect=Exception("API error")):
        info = DataFetcher.get_stock_info("AAPL")
        assert info is None
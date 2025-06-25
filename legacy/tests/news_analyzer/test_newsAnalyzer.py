import pytest, datetime
from unittest.mock import patch, MagicMock
from app import news_analyzer
from app import appconfig
import pytz

TIME_ZONE = pytz.timezone(appconfig.TIME_ZONE)

def test_get_industry_and_sector_returns_dict():
    with patch("app.news_analyzer.data_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {"sector": "Tech", "industry": "Software"}
        result = news_analyzer.get_industry_and_sector("AAPL")
        assert result == {"sector": "Tech", "industry": "Software"}

def test_fetch_recent_news_returns_list():
    with patch("app.news_analyzer.data_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = [
            {"content": {"pubDate":  datetime.datetime.now(TIME_ZONE).strftime('%Y-%m-%dT%H:%M:%SZ'),}}
        ]
        result = news_analyzer.fetch_recent_news("AAPL", days_back=7)
        assert isinstance(result, list)
        assert len(result) == 1

def test_analyze_news_sentiment_calls_processor(monkeypatch):
    called = {}
    class DummyProcessor:
        def _analyze_news_sentiments(self, news_items):
            called['yes'] = True
            return [{"sentiment_score": 0.5}]
    monkeypatch.setattr(news_analyzer, "get_news_processor", lambda: DummyProcessor())
    result = news_analyzer.analyze_news_sentiment([{"content": {"title": "A"}}])
    assert called.get('yes')
    assert isinstance(result, list)
    assert result[0]["sentiment_score"] == 0.5

def test_store_industry_and_news_calls_db(monkeypatch):
    called = {}
    class DummyDB:
        def store_stock_info(self, stock_id, sector, industry):
            called['info'] = (stock_id, sector, industry)
        def store_news_sentiments(self, stock_id, news_sentiments):
            called['news'] = (stock_id, news_sentiments)
    monkeypatch.setattr(news_analyzer, "db_manager", DummyDB())
    news_analyzer.store_industry_and_news("AAPL", 1, {"sector": "Tech", "industry": "Software"}, [{"sentiment_score": 0.5}])
    assert called['info'] == (1, "Tech", "Software")
    assert called['news'][0] == 1

def test_get_average_sentiment_calls_db(monkeypatch):
    class DummyDB:
        def get_average_sentiment(self, stock_id, days_back):
            return 0.42
    monkeypatch.setattr(news_analyzer, "db_manager", DummyDB())
    avg = news_analyzer.get_average_sentiment(1, 7)
    assert avg == 0.42

def test_get_sentiment_timeseries_calls_db(monkeypatch):
    class DummyDB:
        def get_sentiment_timeseries(self, stock_id, days_back):
            return "timeseries"
    monkeypatch.setattr(news_analyzer, "db_manager", DummyDB())
    ts = news_analyzer.get_sentiment_timeseries(1, 60)
    assert ts == "timeseries"

def test_get_industry_average_momentum_calls_analyzer(monkeypatch):
    class DummyAnalyzer:
        def get_industry_average_momentum(self, industry, exclude_stock_id, df_all):
            return 1.23
    monkeypatch.setattr(news_analyzer, "get_industry_analyzer", lambda: DummyAnalyzer())
    val = news_analyzer.get_industry_average_momentum("Software", 1, "df")
    assert val == 1.23

def test_process_stock_news_calls_processor(monkeypatch):
    called = {}
    class DummyProcessor:
        def process_news_for_stock(self, ticker, stock_id, days_back, analyze_sentiment=True):
            called['args'] = (ticker, stock_id, days_back, analyze_sentiment)
    monkeypatch.setattr(news_analyzer, "get_news_processor", lambda: DummyProcessor())
    news_analyzer.process_stock_news("AAPL", 1, 5)
    assert called['args'] == ("AAPL", 1, 5, True)
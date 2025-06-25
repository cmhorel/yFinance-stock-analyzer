import pytest
from app.news_analyzer import NewsProcessor
from app.news_analyzer.data_fetcher import DataFetcher

def test_analyze_news_sentiments_returns_scores():
    news_items = [
        {"content": {"title": "A", "summary": "B"}},
        {"content": {"title": "C", "summary": "D"}}
    ]
    # Patch sentiment analyzer
    class DummyAnalyzer:
        def analyze(self, text):
            return 0.5
    processor = NewsProcessor()
    processor.sentiment_analyzer = DummyAnalyzer()
    results = processor._analyze_news_sentiments(news_items)
    assert isinstance(results, list)
    assert all("sentiment_score" in item for item in results)

def test_process_news_for_stock_calls_fetch(monkeypatch):
    called = {}
    class DummyFetcher:
        def fetch_recent_news(self, ticker, days_back):
            called["yes"] = True
            return [{"content": {"title": "A"}}]
        def get_stock_info(self, ticker):
            return None  # or a dummy dict if needed
    class DummyAnalyzer:
        def analyze(self, text):
            return 0.5
    processor = NewsProcessor()
    processor.sentiment_analyzer = DummyAnalyzer()
    # Patch DataFetcher methods for all instances
    monkeypatch.setattr(DataFetcher, "fetch_recent_news", DummyFetcher().fetch_recent_news)
    monkeypatch.setattr(DataFetcher, "get_stock_info", DummyFetcher().get_stock_info)
    processor.process_news_for_stock("AAPL", 1, 7)
    assert called.get("yes")
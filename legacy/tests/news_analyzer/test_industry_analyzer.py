import pytest
import pandas as pd
from app.news_analyzer import industry_analyzer

class DummyDB:
    def get_industry_stocks(self, industry, exclude_stock_id=None):
        return [1, 2]
    def get_stock_data(self, stock_id, days_back):
        return pd.DataFrame({"close": [100, 101, 102]})

def test_get_industry_average_momentum(monkeypatch):
    analyzer = industry_analyzer.IndustryAnalyzer()
    analyzer.db = DummyDB()
    df_all = pd.DataFrame({
        "stock_id": [1, 2, 3],
        "close": [100, 101, 102],
        "date": pd.date_range("2024-01-01", periods=3)
    })
    result = analyzer.get_industry_average_momentum("Software", 3, df_all)
    assert isinstance(result, float)
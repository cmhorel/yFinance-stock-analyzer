import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from app import stockAnalyzer

def make_test_df():
    # Minimal DataFrame for a ticker
    dates = pd.date_range("2024-01-01", periods=60)
    return pd.DataFrame({
        "date": dates,
        "close": [100 + i for i in range(60)],
        "volume": [1000 + 10*i for i in range(60)],
        "stock_id": [1]*60,
        "sector": ["Technology"]*60,
        "industry": ["Software"]*60,
        "symbol": ["AAPL"]*60
    })

@patch("app.stockAnalyzer.db_manager.get_average_sentiment", return_value=0.2)
@patch("app.stockAnalyzer.news_analyzer.get_industry_average_momentum", return_value=1.0)
def test_analyze_ticker_buy_sell_scores(mock_ind_mom, mock_sent):
    df = make_test_df()
    df_all = df.copy()
    result = stockAnalyzer.analyze_ticker(df, df_all)
    assert isinstance(result, dict)
    assert "buy_score" in result
    assert "sell_score" in result
    assert result["avg_sentiment"] == 0.2

def test_calculate_rsi_typical():
    import numpy as np
    series = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    rsi = stockAnalyzer.calculate_rsi(series)
    assert isinstance(rsi, pd.Series)
    assert (rsi >= 0).all() and (rsi <= 100).all()

def test_get_sector_color_known():
    color = stockAnalyzer.get_sector_color("Technology")
    assert color == "#1f77b4"

def test_get_sector_color_unknown():
    color = stockAnalyzer.get_sector_color("NonexistentSector")
    assert color == stockAnalyzer.SECTOR_COLORS["Unknown"]

@patch("app.stockAnalyzer.go.Figure.write_html")
def test_plot_stock_analysis_saves_html(mock_write_html):
    df = make_test_df()
    with patch("app.stockAnalyzer.news_analyzer.get_average_sentiment", return_value=0.5):
        fig = stockAnalyzer.plot_stock_analysis(df, "AAPL", save_path="plots")
        assert mock_write_html.called
        assert fig is not None

@patch("app.stockAnalyzer.px.scatter")
def test_create_sector_overview_plot_returns_fig(mock_px_scatter):
    mock_fig = MagicMock()
    mock_px_scatter.return_value = mock_fig
    buy = [("AAPL", 6, 0.2, "Software", "Technology")]
    sell = [("TSLA", 5, -0.3, "Auto", "Industrials")]
    fig = stockAnalyzer.create_sector_overview_plot(buy, sell, save_path="plots")
    assert mock_px_scatter.called
    assert fig == mock_fig

@patch("app.stockAnalyzer.db_manager.get_stock_data")
def test_main_prints_and_returns(mock_get_stock_data, capsys):
    # Provide fake data for main
    df = make_test_df()
    mock_get_stock_data.return_value = df
    with patch("app.stockAnalyzer.analyze_ticker", side_effect=lambda g, df_all: {
        "buy_score": 6, "sell_score": 2, "avg_sentiment": 0.5, "industry": "Software", "sector": "Technology"
    }), patch("app.stockAnalyzer.plot_stock_analysis"), patch("app.stockAnalyzer.create_sector_overview_plot"):
        buy, sell = stockAnalyzer.main()
        captured = capsys.readouterr()
        assert "Buy recommendations:" in captured.out
        assert isinstance(buy, list)
        assert "AAPL" in buy

def test_main_no_data(capsys):
    with patch("app.stockAnalyzer.get_stock_data", return_value=pd.DataFrame()):
        result = stockAnalyzer.main()
        captured = capsys.readouterr()
        assert "No data retrieved." in captured.out
        assert result is None

def test_analyze_ticker_includes_volatility():
    df = make_test_df()
    # Add some volatility to the close prices
    df['close'] = [100, 110, 90, 120, 80, 130, 70, 140, 60, 150] * 6
    df_all = df.copy()
    result = stockAnalyzer.analyze_ticker(df, df_all)
    assert "volatility" in result
    assert isinstance(result["volatility"], float)
    # Optionally, check that volatility is above a threshold
    assert result["volatility"] > 0
    
def test_volatility_affects_buy_score():
    # Low volatility: prices barely change
    df_low = make_test_df()
    df_low['close'] = [100] * len(df_low)
    df_all = df_low.copy()
    result_low = stockAnalyzer.analyze_ticker(df_low, df_all)
    
    # High volatility: prices swing up and down
    df_high = make_test_df()
    df_high['close'] = [100 + ((-1) ** i) * 20 for i in range(len(df_high))]
    df_all_high = df_high.copy()
    result_high = stockAnalyzer.analyze_ticker(df_high, df_all_high)
    
    # Assert volatility is higher for the high-volatility input
    assert result_high["volatility"] > result_low["volatility"]
    
    # Assert buy_score is lower for high volatility (if that's your logic)
    assert result_high["buy_score"] < result_low["buy_score"]
# stockAnalyzer.py
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from app import appconfig, news_analyzer
import app.appconfig as appconfig
import os
from app.database_manager import db_manager  # NEW: Import centralized database manager

# Define sector color mapping
SECTOR_COLORS = {
    'Technology': '#1f77b4',
    'Healthcare': '#ff7f0e', 
    'Financial Services': '#2ca02c',
    'Consumer Cyclical': '#d62728',
    'Communication Services': '#9467bd',
    'Industrials': '#8c564b',
    'Consumer Defensive': '#e377c2',
    'Energy': '#7f7f7f',
    'Utilities': '#bcbd22',
    'Real Estate': '#17becf',
    'Basic Materials': '#ff9896',
    'Unknown': '#c7c7c7'
}

def get_sector_color(sector):
    """Get color for a given sector."""
    return SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])

def plot_stock_analysis(df_ticker, ticker, save_path=os.path.join(appconfig.PLOTS_PATH, "stock_analysis")):
    os.makedirs(save_path, exist_ok=True)

    df_ticker = df_ticker.copy()
    df_ticker['MA20'] = df_ticker['close'].rolling(window=20).mean()
    df_ticker['MA50'] = df_ticker['close'].rolling(window=50).mean()
    df_ticker['RSI'] = calculate_rsi(df_ticker['close'])
    df_ticker['Volatility'] = calculate_volatility(df_ticker['close'])  # NEW: Add volatility

    stock_id = df_ticker['stock_id'].iloc[0]
    sector = df_ticker['sector'].iloc[0] if 'sector' in df_ticker.columns and pd.notna(df_ticker['sector'].iloc[0]) else 'Unknown'
    industry = df_ticker['industry'].iloc[0] if 'industry' in df_ticker.columns and pd.notna(df_ticker['industry'].iloc[0]) else 'Unknown'
    
    avg_sentiment = news_analyzer.get_average_sentiment(stock_id)
    sector_color = get_sector_color(sector)

    # Create subplots - Updated to include volatility
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(f'{ticker} - {sector} ({industry})', 'RSI', 'Volatility', 'Volume'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    # Price and Moving Averages
    fig.add_trace(
        go.Scatter(
            x=df_ticker['date'], 
            y=df_ticker['close'],
            mode='lines',
            name='Close Price',
            line=dict(color=sector_color, width=2),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_ticker['date'], 
            y=df_ticker['MA20'],
            mode='lines',
            name='20-day MA',
            line=dict(color='orange', width=1),
            hovertemplate='<b>20-day MA</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_ticker['date'], 
            y=df_ticker['MA50'],
            mode='lines',
            name='50-day MA',
            line=dict(color='green', width=1),
            hovertemplate='<b>50-day MA</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

     # RSI
    fig.add_trace(
        go.Scatter(
            x=df_ticker['date'], 
            y=df_ticker['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=1),
            hovertemplate='<b>RSI</b><br>Date: %{x}<br>RSI: %{y:.1f}<extra></extra>'
        ),
        row=2, col=1
    )

    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # NEW: Volatility
    fig.add_trace(
        go.Scatter(
            x=df_ticker['date'], 
            y=df_ticker['Volatility'],
            mode='lines',
            name='Volatility',
            line=dict(color='red', width=1),
            hovertemplate='<b>Volatility</b><br>Date: %{x}<br>Volatility: %{y:.3f}<extra></extra>'
        ),
        row=3, col=1
    )

    # Volume
    fig.add_trace(
        go.Bar(
            x=df_ticker['date'], 
            y=df_ticker['volume'],
            name='Volume',
            marker_color='lightgray',
            hovertemplate='<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
        ),
        row=4, col=1
    )

    # ... existing code for sentiment annotation ...

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Analysis - {sector} Sector',
        xaxis_title='Date',
        height=900,  # Increased height for additional subplot
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volatility", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    # Add sentiment annotation
    sentiment_color = 'green' if avg_sentiment > 0 else 'red' if avg_sentiment < 0 else 'gray'
    sentiment_text = f'Avg Sentiment: {avg_sentiment:.3f}'
    
    fig.add_annotation(
        x=df_ticker['date'].iloc[-1],
        y=df_ticker['close'].max(),
        text=sentiment_text,
        showarrow=True,
        arrowhead=2,
        arrowcolor=sentiment_color,
        bgcolor="white",
        bordercolor=sentiment_color,
        borderwidth=2,
        font=dict(color=sentiment_color, size=12),
        row=1, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Analysis - {sector} Sector',
        xaxis_title='Date',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=3, col=1)

    # Save as HTML
    filename = os.path.join(save_path, f"{ticker}_analysis.html")
    fig.write_html(filename)
    print(f"Saved interactive plot to {filename}")

    return fig



def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Neutral RSI for initial points

def get_stock_data(conn=None, months_back=6):  # MODIFIED: Make conn optional
    """Get stock data using centralized database manager."""
    return db_manager.get_stock_data(months_back)


def analyze_ticker(df_ticker, df_all):  # NEW: Pass df_all for industry comparisons
    # Require enough data points
    if len(df_ticker) < 20:
        return None

    stock_id = df_ticker['stock_id'].iloc[0]
    industry = df_ticker['industry'].iloc[0] if 'industry' in df_ticker.columns and pd.notna(df_ticker['industry'].iloc[0]) else 'Unknown'
    stock_id = df_ticker['stock_id'].iloc[0]
    industry = df_ticker['industry'].iloc[0] if 'industry' in df_ticker.columns and pd.notna(df_ticker['industry'].iloc[0]) else 'Unknown'
    sector = df_ticker['sector'].iloc[0] if 'sector' in df_ticker.columns and pd.notna(df_ticker['sector'].iloc[0]) else 'Unknown'

    close = df_ticker['close']
    volume = df_ticker['volume']

    # Moving averages
    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()

    # RSI
    rsi = calculate_rsi(close)

    # NEW: Volatility
    volatility = calculate_volatility(close)

    # Momentum: difference between current close and 7 days ago
    momentum = close - close.shift(7)

    # Volume change: compare avg volume last 5 days to previous 5 days
    vol_change = volume.rolling(window=5).mean() - volume.rolling(window=5).mean().shift(5)

    # NEW: Get average news sentiment from last 7 days
    avg_sentiment = db_manager.get_average_sentiment(stock_id)

    # NEW: Get industry-average momentum for relative comparison
    industry_avg_momentum = news_analyzer.get_industry_average_momentum(industry, stock_id, df_all) if industry != 'Unknown' else 0.0

    latest_idx = df_ticker.index[-1]
    metrics = {
        'close': close.iloc[-1],
        'ma20': ma20.iloc[-1],
        'ma50': ma50.iloc[-1],
        'rsi': rsi.iloc[-1],
        'momentum': momentum.iloc[-1],
        'vol_change': vol_change.iloc[-1],
        'avg_sentiment': avg_sentiment,
        'volatility': volatility.iloc[-1],  # NEW: Add volatility metric
    }
    metrics['relative_momentum'] = metrics.get('momentum', 0) - industry_avg_momentum  # NEW: Relative to industry

    # Buy if price above MAs, RSI low, momentum and volume increasing, volatility reasonable
    buy_score = 0
    buy_score += 1 if metrics['close'] > metrics['ma20'] else 0
    buy_score += 1 if metrics['close'] > metrics['ma50'] else 0
    buy_score += 1 if metrics['rsi'] < 40 else 0
    buy_score += 1 if metrics['momentum'] > 0 else 0
    buy_score += 1 if metrics['vol_change'] > 0 else 0
    # NEW: Sentiment and industry factors
    buy_score += 1 if metrics['avg_sentiment'] > 0.1 else 0  # Boost for positive news
    buy_score += 1 if metrics['relative_momentum'] > 0 else 0  # Boost if outperforming industry
    # NEW: Volatility factor - prefer moderate volatility (not too high, not too low)
    buy_score += 1 if metrics['volatility'] <= 0.3 else 0

    # Sell if price below MAs, RSI high, momentum and volume decreasing, high volatility
    sell_score = 0
    sell_score += 1 if metrics['close'] < metrics['ma20'] else 0
    sell_score += 1 if metrics['close'] < metrics['ma50'] else 0
    sell_score += 1 if metrics['rsi'] > 60 else 0
    sell_score += 1 if metrics['momentum'] < 0 else 0
    sell_score += 1 if metrics['vol_change'] < 0 else 0
    # NEW: Sentiment and industry factors
    sell_score += 1 if metrics['avg_sentiment'] < -0.1 else 0  # Boost for negative news
    sell_score += 1 if metrics['relative_momentum'] < 0 else 0  # Boost if underperforming industry
    # NEW: High volatility increases sell score
    sell_score += 1 if metrics['volatility'] > 0.5 else 0

    return {
        'buy_score': buy_score, 
        'sell_score': sell_score, 
        'avg_sentiment': avg_sentiment, 
        'industry': industry,
        'sector': sector,
        'volatility': metrics['volatility']  # NEW: Include volatility in return
    }


def calculate_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate rolling volatility (standard deviation of returns)."""
    returns = series.pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized volatility
    return volatility.fillna(0)

def create_sector_overview_plot(buy_candidates, sell_candidates, save_path=appconfig.PLOTS_PATH):
    """Create an interactive sector overview plot."""
    os.makedirs(save_path, exist_ok=True)
    
    # Combine all candidates
    all_candidates = []
    for ticker, score, sentiment, industry, sector in buy_candidates:
        all_candidates.append({
            'ticker': ticker,
            'score': score,
            'sentiment': sentiment,
            'industry': industry,
            'sector': sector,
            'recommendation': 'Buy'
        })
    
    for ticker, score, sentiment, industry, sector in sell_candidates:
        all_candidates.append({
            'ticker': ticker,
            'score': score,
            'sentiment': sentiment,
            'industry': industry,
            'sector': sector,
            'recommendation': 'Sell'
        })
    
    if not all_candidates:
        return
    
    df_candidates = pd.DataFrame(all_candidates)
    
    # Create scatter plot
    fig = px.scatter(
        df_candidates,
        x='sentiment',
        y='score',
        color='sector',
        symbol='recommendation',
        size='score',
        hover_data=['ticker', 'industry'],
        title='Stock Recommendations by Sector and Sentiment',
        labels={
            'sentiment': 'Average Sentiment Score',
            'score': 'Recommendation Score',
            'sector': 'Sector'
        },
        color_discrete_map=SECTOR_COLORS
    )
    
    fig.update_layout(
        height=600,
        showlegend=True
    )
    
    filename = os.path.join(save_path, "sector_overview.html")
    fig.write_html(filename)
    print(f"Saved sector overview plot to {filename}")
    
    return fig

def main():
    df = get_stock_data()
    if df.empty:
        print("No data retrieved.")
        return

    buy_candidates = []
    sell_candidates = []

    grouped = df.groupby('symbol')

    for ticker, group in grouped:
        result = analyze_ticker(group, df)
        if result is None:
            continue
        
        if result['buy_score'] >= 5:
            buy_candidates.append((ticker, result['buy_score'], result['avg_sentiment'], result['industry'], result['sector']))
        if result['sell_score'] >= 5:
            sell_candidates.append((ticker, result['sell_score'], result['avg_sentiment'], result['industry'], result['sector']))

    buy_candidates.sort(key=lambda x: x[1], reverse=True)
    sell_candidates.sort(key=lambda x: x[1], reverse=True)

    buy_tickers = [t[0] for t in buy_candidates[:10]]
    sell_tickers = [t[0] for t in sell_candidates[:10]]

    # Enhanced output with sector information
    print("Buy recommendations:")
    for t in buy_candidates[:10]:
        print(f"{t[0]} (Score: {t[1]}, Sentiment: {t[2]:.3f}, Sector: {t[4]}, Industry: {t[3]})")
    print("Sell recommendations:")
    for t in sell_candidates[:10]:
        print(f"{t[0]} (Score: {t[1]}, Sentiment: {t[2]:.3f}, Sector: {t[4]}, Industry: {t[3]})")

    # Create sector overview plot
    create_sector_overview_plot(buy_candidates, sell_candidates)

    # Plot individual stock analyses
    for ticker, group in grouped:
        df_ticker = grouped.get_group(ticker)
        plot_stock_analysis(df_ticker, ticker)

    return buy_tickers, sell_tickers


if __name__ == "__main__":
    main()

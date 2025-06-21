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
    df_ticker['Volatility'] = calculate_volatility(df_ticker['close'])
    
    # Calculate buy/sell signals for visualization
    df_ticker['Buy_Signal'] = generate_buy_sell_signals(df_ticker, signal_type='buy')
    df_ticker['Sell_Signal'] = generate_buy_sell_signals(df_ticker, signal_type='sell')

    stock_id = df_ticker['stock_id'].iloc[0]
    sector = df_ticker['sector'].iloc[0] if 'sector' in df_ticker.columns and pd.notna(df_ticker['sector'].iloc[0]) else 'Unknown'
    industry = df_ticker['industry'].iloc[0] if 'industry' in df_ticker.columns and pd.notna(df_ticker['industry'].iloc[0]) else 'Unknown'
    
    avg_sentiment = news_analyzer.get_average_sentiment(stock_id)
    sector_color = get_sector_color(sector)

    # Create subplots with volatility included
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

    # RSI reference lines - add as shapes instead
    fig.add_shape(
        type="line", x0=df_ticker['date'].iloc[0], x1=df_ticker['date'].iloc[-1],
        y0=70, y1=70, line=dict(color="red", width=1, dash="dash"),
        row=2, col=1
    )
    fig.add_shape(
        type="line", x0=df_ticker['date'].iloc[0], x1=df_ticker['date'].iloc[-1],
        y0=30, y1=30, line=dict(color="green", width=1, dash="dash"),
        row=2, col=1
    )

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

    # Add buy/sell signals to the price chart
    buy_signals = df_ticker[df_ticker['Buy_Signal'] == 1]
    sell_signals = df_ticker[df_ticker['Sell_Signal'] == 1]
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['date'], 
                y=buy_signals['close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                hovertemplate='<b>Buy Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['date'], 
                y=sell_signals['close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                hovertemplate='<b>Sell Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

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

    # Update layout - single layout update
    fig.update_layout(
        title=f'{ticker} Stock Analysis - {sector} Sector',
        xaxis_title='Date',
        height=900,  # Increased height for 4 subplots
        showlegend=True,
        hovermode='x unified'
    )

    # Update y-axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volatility", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)

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

def generate_buy_sell_signals(df_ticker, signal_type='buy'):
    """Generate buy/sell signals for visualization on charts."""
    if len(df_ticker) < 50:
        return pd.Series([0] * len(df_ticker), index=df_ticker.index)
    
    close = df_ticker['close']
    volume = df_ticker['volume']
    
    # Calculate indicators
    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()
    rsi = calculate_rsi(close)
    volatility = calculate_volatility(close)
    
    signals = pd.Series([0] * len(df_ticker), index=df_ticker.index)
    
    for i in range(50, len(df_ticker)):  # Start after enough data for indicators
        current_close = close.iloc[i]
        current_ma20 = ma20.iloc[i]
        current_ma50 = ma50.iloc[i]
        current_rsi = rsi.iloc[i]
        current_vol = volatility.iloc[i]
        
        # 7-day momentum
        momentum_7d = (close.iloc[i] - close.iloc[i-7]) / close.iloc[i-7] if i >= 7 else 0
        
        # Volume surge
        avg_volume = volume.iloc[i-20:i].mean()
        volume_ratio = volume.iloc[i] / avg_volume if avg_volume > 0 else 1
        
        if signal_type == 'buy':
            # Buy signal conditions (more conservative)
            conditions = [
                current_close > current_ma20,  # Price above 20-day MA
                current_ma20 > current_ma50,   # 20-day MA above 50-day MA (uptrend)
                current_rsi < 40,              # RSI oversold
                momentum_7d > 0.02,            # 2%+ weekly momentum
                volume_ratio > 1.3,            # Volume surge
                current_vol < 0.4              # Reasonable volatility
            ]
            
            if sum(conditions) >= 4:  # Require at least 4 conditions
                signals.iloc[i] = 1
                
        elif signal_type == 'sell':
            # Sell signal conditions
            conditions = [
                current_close < current_ma20,  # Price below 20-day MA
                current_ma20 < current_ma50,   # 20-day MA below 50-day MA (downtrend)
                current_rsi > 70,              # RSI overbought
                momentum_7d < -0.02,           # 2%+ weekly decline
                current_vol > 0.5              # High volatility
            ]
            
            if sum(conditions) >= 3:  # Require at least 3 conditions
                signals.iloc[i] = 1
    
    return signals

def create_sector_overview_plot(buy_candidates, sell_candidates, save_path=appconfig.PLOTS_PATH):
    """Create an enhanced interactive sector overview plot with volatility information."""
    os.makedirs(save_path, exist_ok=True)
    
    # Get full stock data for volatility calculation
    df = get_stock_data()
    grouped = df.groupby('symbol')
    
    # Combine all candidates with enhanced data
    all_candidates = []
    for ticker, score, sentiment, industry, sector in buy_candidates:
        # Get volatility for this ticker
        volatility = 0.2  # default
        if ticker in grouped.groups:
            ticker_data = grouped.get_group(ticker)
            if len(ticker_data) > 20:
                vol_series = calculate_volatility(ticker_data['close'])
                volatility = vol_series.iloc[-1] if not pd.isna(vol_series.iloc[-1]) else 0.2
        
        all_candidates.append({
            'ticker': ticker,
            'score': score,
            'sentiment': sentiment,
            'industry': industry,
            'sector': sector,
            'volatility': volatility,
            'recommendation': 'Buy',
            'risk_level': 'Low' if volatility < 0.3 else 'Medium' if volatility < 0.5 else 'High'
        })
    
    for ticker, score, sentiment, industry, sector in sell_candidates:
        # Get volatility for this ticker
        volatility = 0.2  # default
        if ticker in grouped.groups:
            ticker_data = grouped.get_group(ticker)
            if len(ticker_data) > 20:
                vol_series = calculate_volatility(ticker_data['close'])
                volatility = vol_series.iloc[-1] if not pd.isna(vol_series.iloc[-1]) else 0.2
        
        all_candidates.append({
            'ticker': ticker,
            'score': score,
            'sentiment': sentiment,
            'industry': industry,
            'sector': sector,
            'volatility': volatility,
            'recommendation': 'Sell',
            'risk_level': 'Low' if volatility < 0.3 else 'Medium' if volatility < 0.5 else 'High'
        })
    
    if not all_candidates:
        return
    
    df_candidates = pd.DataFrame(all_candidates)
    
    # Create enhanced scatter plot with volatility as size
    fig = px.scatter(
        df_candidates,
        x='sentiment',
        y='score',
        color='sector',
        symbol='recommendation',
        size='volatility',
        hover_data=['ticker', 'industry', 'volatility', 'risk_level'],
        title='Stock Recommendations: Sentiment vs Score (Size = Volatility)',
        labels={
            'sentiment': 'Average Sentiment Score',
            'score': 'Recommendation Score',
            'sector': 'Sector',
            'volatility': 'Volatility'
        },
        color_discrete_map=SECTOR_COLORS,
        size_max=20
    )
    
    # Add quadrant lines for better interpretation
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant annotations
    fig.add_annotation(x=0.3, y=8, text="High Score<br>Positive Sentiment", 
                      showarrow=False, bgcolor="lightgreen", opacity=0.7)
    fig.add_annotation(x=-0.3, y=8, text="High Score<br>Negative Sentiment", 
                      showarrow=False, bgcolor="lightyellow", opacity=0.7)
    fig.add_annotation(x=0.3, y=3, text="Low Score<br>Positive Sentiment", 
                      showarrow=False, bgcolor="lightblue", opacity=0.7)
    fig.add_annotation(x=-0.3, y=3, text="Low Score<br>Negative Sentiment", 
                      showarrow=False, bgcolor="lightcoral", opacity=0.7)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_title="News Sentiment Score (Negative ← → Positive)",
        yaxis_title="Recommendation Score (Higher = Stronger Signal)"
    )
    
    filename = os.path.join(save_path, "sector_overview.html")
    fig.write_html(filename)
    print(f"Saved enhanced sector overview plot to {filename}")
    
    # Also create a volatility-focused chart
    create_volatility_risk_chart(df_candidates, save_path)
    
    return fig

def create_volatility_risk_chart(df_candidates, save_path):
    """Create a separate chart focusing on volatility and risk analysis."""
    
    # Create a copy and fix sentiment for size (must be positive)
    df_viz = df_candidates.copy()
    df_viz['sentiment_abs'] = abs(df_viz['sentiment']) + 0.1  # Add small offset to avoid zero size
    
    # Create volatility vs score chart
    fig = px.scatter(
        df_viz,
        x='volatility',
        y='score',
        color='recommendation',
        size='sentiment_abs',
        hover_data=['ticker', 'sector', 'industry', 'risk_level', 'sentiment'],
        title='Risk Analysis: Volatility vs Recommendation Score',
        labels={
            'volatility': 'Annualized Volatility',
            'score': 'Recommendation Score',
            'sentiment_abs': 'Sentiment Strength'
        },
        color_discrete_map={'Buy': 'green', 'Sell': 'red'}
    )
    
    # Add risk zones
    fig.add_vrect(x0=0, x1=0.3, fillcolor="green", opacity=0.1, 
                  annotation_text="Low Risk", annotation_position="top left")
    fig.add_vrect(x0=0.3, x1=0.5, fillcolor="yellow", opacity=0.1, 
                  annotation_text="Medium Risk", annotation_position="top")
    fig.add_vrect(x0=0.5, x1=2, fillcolor="red", opacity=0.1, 
                  annotation_text="High Risk", annotation_position="top right")
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_title="Volatility (Risk Level)",
        yaxis_title="Recommendation Score"
    )
    
    filename = os.path.join(save_path, "volatility_risk_analysis.html")
    fig.write_html(filename)
    print(f"Saved volatility risk analysis to {filename}")
    
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

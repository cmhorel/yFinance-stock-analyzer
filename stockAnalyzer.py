import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import newsAnalyzer  # New import for sentiment and industry functions
import config  # Assuming config.py contains DB_NAME

def plot_stock_analysis(df_ticker, ticker, save_path='plots'):
    import os
    os.makedirs(save_path, exist_ok=True)

    df_ticker = df_ticker.copy()
    df_ticker['MA20'] = df_ticker['close'].rolling(window=20).mean()
    df_ticker['MA50'] = df_ticker['close'].rolling(window=50).mean()
    df_ticker['RSI'] = calculate_rsi(df_ticker['close'])

    stock_id = df_ticker['stock_id'].iloc[0]
    avg_sentiment = newsAnalyzer.get_average_sentiment(stock_id)
    print(avg_sentiment)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Price and Moving Averages
    ax1.plot(df_ticker['date'], df_ticker['close'], label='Close Price', color='blue')
    ax1.plot(df_ticker['date'], df_ticker['MA20'], label='20-day MA', color='orange')
    ax1.plot(df_ticker['date'], df_ticker['MA50'], label='50-day MA', color='green')
    ax1.set_title(f'{ticker} Price with Moving Averages and Sentiment')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # RSI overlay
    ax1_rsi = ax1.twinx()
    ax1_rsi.plot(df_ticker['date'], df_ticker['RSI'], label='RSI', color='purple', linestyle='--')
    ax1_rsi.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax1_rsi.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax1_rsi.set_ylabel('RSI')
    ax1_rsi.legend(loc='upper right')

    # Volume bars
    ax2.bar(df_ticker['date'], df_ticker['volume'], color='gray')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True)

    # Add sentiment annotation on the far right
    sentiment_color = 'green' if avg_sentiment > 0 else 'red' if avg_sentiment < 0 else 'gray'
    sentiment_text = f'Sentiment: {avg_sentiment:.2f}'
    max_y = df_ticker['close'].max()
    min_y = df_ticker['close'].min()
    y_pos = max_y - (max_y - min_y) * 0.1  # 10% down from the top

    ax1.annotate(sentiment_text, 
                 xy=(df_ticker['date'].iloc[-1], y_pos),
                 xytext=(df_ticker['date'].iloc[-1], y_pos),
                 fontsize=12, color=sentiment_color,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=sentiment_color, lw=2))

    plt.tight_layout()
    filename = os.path.join(save_path, f"{ticker}_analysis.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot to {filename}")


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Neutral RSI for initial points


def get_stock_data(conn, months_back=6):
    cutoff_date = datetime.now() - timedelta(days=months_back * 30)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')

    query = f"""
    SELECT st.id AS stock_id, st.symbol, sp.date, sp.close, sp.volume,
           si.industry 
    FROM stock_prices sp
    JOIN stocks st ON sp.stock_id = st.id
    LEFT JOIN stock_info si ON st.id = si.stock_id  
    WHERE sp.date >= ?
    ORDER BY st.symbol, sp.date
    """
    try:
        df = pd.read_sql_query(query, conn, params=(cutoff_str,))
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"SQL query error: {e}")
        return pd.DataFrame()



def analyze_ticker(df_ticker, df_all):  # NEW: Pass df_all for industry comparisons
    # Require enough data points
    if len(df_ticker) < 20:
        return None

    stock_id = df_ticker['stock_id'].iloc[0]
    industry = df_ticker['industry'].iloc[0] if 'industry' in df_ticker.columns and pd.notna(df_ticker['industry'].iloc[0]) else 'Unknown'


    close = df_ticker['close']
    volume = df_ticker['volume']

    # Moving averages
    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()

    # RSI
    rsi = calculate_rsi(close)

    # Momentum: difference between current close and 7 days ago
    momentum = close - close.shift(7)

    # Volume change: compare avg volume last 5 days to previous 5 days
    vol_change = volume.rolling(window=5).mean() - volume.rolling(window=5).mean().shift(5)



    # NEW: Get average news sentiment from last 7 days
    avg_sentiment = newsAnalyzer.get_average_sentiment(stock_id)

    # NEW: Get industry-average momentum for relative comparison
    industry_avg_momentum = newsAnalyzer.get_industry_average_momentum(industry, stock_id, df_all) if industry != 'Unknown' else 0.0

    latest_idx = df_ticker.index[-1]
    metrics = {
        'close': close.iloc[-1],
        'ma20': ma20.iloc[-1],
        'ma50': ma50.iloc[-1],
        'rsi': rsi.iloc[-1],
        'momentum': momentum.iloc[-1],
        'vol_change': vol_change.iloc[-1],
        'avg_sentiment': avg_sentiment,
        
    }
    print(metrics)
    metrics['relative_momentum'] = metrics.get('momentum', 0) - industry_avg_momentum  # NEW: Relative to industry

    # Buy if price above MAs, RSI low, momentum and volume increasing
    buy_score = 0
    buy_score += 1 if metrics['close'] > metrics['ma20'] else 0
    buy_score += 1 if metrics['close'] > metrics['ma50'] else 0
    buy_score += 1 if metrics['rsi'] < 40 else 0
    buy_score += 1 if metrics['momentum'] > 0 else 0
    buy_score += 1 if metrics['vol_change'] > 0 else 0
    # NEW: Sentiment and industry factors
    buy_score += 1 if metrics['avg_sentiment'] > 0.1 else 0  # Boost for positive news
    buy_score += 1 if metrics['relative_momentum'] > 0 else 0  # Boost if outperforming industry

    # Sell if price below MAs, RSI high, momentum and volume decreasing
    sell_score = 0
    sell_score += 1 if metrics['close'] < metrics['ma20'] else 0
    sell_score += 1 if metrics['close'] < metrics['ma50'] else 0
    sell_score += 1 if metrics['rsi'] > 60 else 0
    sell_score += 1 if metrics['momentum'] < 0 else 0
    sell_score += 1 if metrics['vol_change'] < 0 else 0
    # NEW: Sentiment and industry factors
    sell_score += 1 if metrics['avg_sentiment'] < -0.1 else 0  # Boost for negative news
    sell_score += 1 if metrics['relative_momentum'] < 0 else 0  # Boost if underperforming industry

    return {'buy_score': buy_score, 'sell_score': sell_score, 'avg_sentiment': avg_sentiment, 'industry': industry}


def main():
    conn = sqlite3.connect(config.DB_NAME)
    df = get_stock_data(conn)
    if df.empty:
        print("No data retrieved.")
        return

    buy_candidates = []
    sell_candidates = []

    grouped = df.groupby('symbol')

    for ticker, group in grouped:
        result = analyze_ticker(group, df)  # NEW: Pass full df for industry comparisons
        if result is None:
            continue
        # NEW: Include sentiment and industry in output for transparency
        if result['buy_score'] >= 5:  # Adjusted threshold to account for new factors (max now 7)
            buy_candidates.append((ticker, result['buy_score'], result['avg_sentiment'], result['industry']))
        if result['sell_score'] >= 5:
            sell_candidates.append((ticker, result['sell_score'], result['avg_sentiment'], result['industry']))

    buy_candidates.sort(key=lambda x: x[1], reverse=True)
    sell_candidates.sort(key=lambda x: x[1], reverse=True)

    buy_tickers = [t[0] for t in buy_candidates[:10]]
    sell_tickers = [t[0] for t in sell_candidates[:10]]

    # NEW: Enhanced output with sentiment and industry details
    print("Buy recommendations:")
    for t in buy_candidates[:10]:
        print(f"{t[0]} (Score: {t[1]}, Avg Sentiment: {t[2]:.2f}, Industry: {t[3]})")
    print("Sell recommendations:")
    for t in sell_candidates[:10]:
        print(f"{t[0]} (Score: {t[1]}, Avg Sentiment: {t[2]:.2f}, Industry: {t[3]})")

    # Plot fundamentals for each buy and sell ticker
    for ticker in buy_tickers + sell_tickers:
        df_ticker = grouped.get_group(ticker)
        plot_stock_analysis(df_ticker, ticker)

    conn.close()
    return buy_tickers, sell_tickers


if __name__ == "__main__":
    main()

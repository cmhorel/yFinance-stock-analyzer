# newsAnalyzer.py
import sqlite3
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import pandas as pd

DB_NAME = 'stocks.db'

def get_sentiment_timeseries(stock_id, days_back=60):
    """
    Retrieves daily average sentiment scores for the stock over the past 'days_back' days.
    Returns a pandas DataFrame with 'date' and 'sentiment_score' columns.
    """
    conn = sqlite3.connect(DB_NAME)
    query = '''
        SELECT date, AVG(sentiment_score) as avg_sentiment
        FROM stock_news
        WHERE stock_id = ? AND date >= ?
        GROUP BY date
        ORDER BY date ASC
    '''
    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    df_sentiment = pd.read_sql_query(query, conn, params=(stock_id, cutoff_date))
    conn.close()

    if df_sentiment.empty:
        return pd.DataFrame({'date': [], 'avg_sentiment': []})

    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
    return df_sentiment

def get_industry_and_sector(ticker):
    """
    Fetches industry and sector for a ticker using yfinance.
    Returns a dict with 'sector' and 'industry' or None on failure.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown')
        }
    except Exception as e:
        print(f"Error fetching industry for {ticker}: {e}")
        return None

def fetch_recent_news(ticker, days_back=7):
    """
    Fetches recent news for the ticker using yfinance.
    Returns a list of news items from the last 'days_back' days.
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        news = yf.Ticker(ticker).news
        recent_news = [
            item for item in news
            if (datetime.strptime(item['content']['pubDate'], '%Y-%m-%dT%H:%M:%SZ') >= cutoff_date)
        ]
        return recent_news
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def analyze_news_sentiment(news_items):
    """
    Analyzes sentiment for a list of news items using TextBlob.
    Returns a list of dicts with title, summary, and sentiment score (-1 to 1).
    """
    sentiments = []
    for item in news_items:
        text = f"{item['content'].get('title', '')} {item['content']['provider'].get('displayName', '')} {item['content'].get('summary', '')}"
        # if not text.strip():
        #     continue
        analysis = TextBlob(text)
        sentiments.append({
            'title': item['content'].get('title', 'N/A'),
            'summary': item['content'].get('summary', 'N/A'),
            'publish_date':datetime.strptime(item['content']['pubDate'], '%Y-%m-%dT%H:%M:%SZ') if 'pubDate' in item['content']  else 'N/A',
            'sentiment_score': analysis.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
        })
    return sentiments

def store_industry_and_news(ticker, stock_id, industry_data, news_sentiments, c):
    """
    Stores industry/sector and news sentiments in the database.
    """
    print(f"sync {stock_id}, {industry_data['sector']}, {industry_data['industry']})")
    try:
        # Store industry and sector in stock_info
        if industry_data:
            c.execute('''
                INSERT OR REPLACE INTO stock_info (stock_id, sector, industry)
                VALUES (?, ?, ?)
            ''', (stock_id, industry_data['sector'], industry_data['industry']))
        
        # Store news sentiments in stock_news
        for sentiment in news_sentiments:
            c.execute('''
                INSERT OR IGNORE INTO stock_news (stock_id, date, title, summary, sentiment_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (stock_id, sentiment['publish_date'], sentiment['title'], sentiment['summary'], sentiment['sentiment_score']))
    except Exception as e:
        print(f"Error storing data for {ticker}: {e}")

def get_average_sentiment(stock_id, days_back=7):
    """
    Retrieves and averages sentiment scores for a stock from the last 'days_back' days.
    Returns the average score or 0 if no data.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    print(f"stock_id: {stock_id} (type: {type(stock_id)})")
    query = f''' SELECT AVG(sentiment_score) FROM stock_news
        WHERE stock_id = {stock_id} AND date >= DATE({cutoff_date})
'''
    c.execute(query)
    # c.execute('''
    #     SELECT AVG(sentiment_score) FROM stock_news
    #     WHERE stock_id = ? AND date >= DATE(?)
    # ''', (stock_id, cutoff_date))
    result = c.fetchall()
    avg_sentiment = result[0][0] or 0.0
    conn.close()
    return avg_sentiment

def get_industry_average_momentum(industry, exclude_stock_id, df_all):
    """
    Computes average momentum for stocks in the same industry, excluding the current stock.
    Uses data from stockAnalyzer's get_stock_data for comparison.
    Returns average momentum or 0 if insufficient data.
    """
    industry_stocks = df_all[df_all['industry'] == industry]['stock_id'].unique()
    industry_stocks = [sid for sid in industry_stocks if sid != exclude_stock_id]
    if not industry_stocks:
        return 0.0
    
    momenta = []
    for sid in industry_stocks:
        df_ticker = df_all[df_all['stock_id'] == sid]
        if len(df_ticker) < 8:  # Need at least 8 points for momentum (close - shift(7))
            continue
        momentum = df_ticker['close'] - df_ticker['close'].shift(7)
        momenta.append(momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0)
    
    return sum(momenta) / len(momenta) if momenta else 0.0
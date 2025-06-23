# stockScraper.py
import sqlite3, time, yfinance as yf, pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import app.appconfig as appconfig
import app.news_analyzer as news_analyzer  # NEW: Import news analyzer module
from app.database_manager import DatabaseManager  # NEW: Import centralized database manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.news_analyzer.sentiment_analyzer import SentimentAnalyzer
import curl_cffi.requests as requests
from yfinance.exceptions import YFRateLimitError
import pytz


TIME_ZONE  = pytz.timezone(appconfig.TIME_ZONE)  # NEW: Use timezone from appconfig

yt_session = requests.Session(impersonate="chrome")

db_manager = DatabaseManager()  # NEW: Create a global instance of the database manager

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DB_NAME = appconfig.DB_NAME
THREADS = 10
BATCH_SIZE = 5

def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Create 'stocks' table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE
        )
    ''')

    # Create 'stock_prices' table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            stock_id INTEGER,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (stock_id, date),
            UNIQUE (stock_id, date),
            FOREIGN KEY (stock_id) REFERENCES stocks(id)
        )
    ''')

    # NEW: Create 'stock_info' table for industry and sector
    c.execute('''
        CREATE TABLE IF NOT EXISTS stock_info (
            stock_id INTEGER PRIMARY KEY,
            sector TEXT,
            industry TEXT,
            FOREIGN KEY (stock_id) REFERENCES stocks(id)
        )
    ''')

    # NEW: Create 'stock_news' table for news and sentiments
    c.execute('''
        CREATE TABLE IF NOT EXISTS stock_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER,
            date TEXT,
            title TEXT,
            summary TEXT,
            sentiment_score REAL,
            UNIQUE (stock_id, date, title, summary), 
            FOREIGN KEY (stock_id) REFERENCES stocks(id)
        )
    ''')

    conn.commit()
    conn.close()



def yf_download_with_retry(*args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            data = yf.download(*args, **kwargs, session=yt_session)
            return data
        except YFRateLimitError as e:
            print("YFinance rate limit error, backing off and retrying...")
            time.sleep(0.05)  # 50ms
            continue

    raise RuntimeError("Max retries exceeded for yfinance download")

def get_latest_date(symbol):
    """Get latest date using database manager."""
    return db_manager.get_latest_stock_price_date(symbol)

def get_nasdaq_100_tickers():
    tables = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')
    tickers = tables[4]['Ticker'].tolist()
    return [t.replace('.', '-') for t in tickers]

def get_tsx60_tickers():
    tables = pd.read_html('https://en.wikipedia.org/wiki/S%26P/TSX_60')
    for table in tables:
        if 'Symbol' in table.columns:
            tickers = table['Symbol'].dropna().tolist()  # Drop NaN rows
            # Clean and format tickers
            return [str(t).replace('.', '-') + '.TO' for t in tickers]
    
    raise ValueError("TSX‑60 'Symbol' table not found")

def sync_batch(tickers):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # First get stock IDs for all tickers, insert if not exists
    stock_ids = {}
    for s in tickers:
        c.execute('INSERT OR IGNORE INTO stocks (symbol) VALUES (?)', (s,))
        c.execute('SELECT id FROM stocks WHERE symbol = ?', (s,))
        stock_ids[s] = c.fetchone()[0]

    starts = {}
    for s in tickers:
        ld = get_latest_date(s)
        starts[s] = ((datetime.strptime(ld,'%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) if ld else '2020-01-01'
    start = min(starts.values())

    try:
        #hist = yf.download(tickers, start=start, group_by='ticker', threads=True, progress=False)
        hist = yf_download_with_retry(tickers, start=start, group_by='ticker', threads=True, progress=False)
        for s in tickers:
            if s not in hist.columns.get_level_values(0):
                continue
            df = hist[s].dropna()
            for idx, row in df.iterrows():
                d = idx.strftime('%Y-%m-%d')
                if d < starts[s]: continue
                c.execute('INSERT OR IGNORE INTO stock_prices VALUES (?,?,?,?,?,?,?)',
                          (stock_ids[s], d, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
        conn.commit()
    except Exception as e:
        print(f"Error syncing {tickers}: {e}")
    finally:
        conn.close()


def plot_all(tickers):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    plt.figure(figsize=(19.2,10.8))
    for s in tickers:
        c.execute('''
            SELECT sp.date, sp.close
            FROM stock_prices sp
            JOIN stocks st ON sp.stock_id = st.id
            WHERE st.symbol = ?
            ORDER BY sp.date
        ''', (s,))
        rows = c.fetchall()
        if not rows: continue
        dates, prices = zip(*[(datetime.strptime(d,'%Y-%m-%d'), p) for d,p in rows])
        plt.plot(dates, prices, linewidth=0.8)
        plt.text(dates[-1], prices[-1], f' {s}', fontsize=8, va='center')
    conn.close()
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('NASDAQ‑100 & TSX‑60 Prices')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def sync_ticker(ticker, analyze_sentiment=True):
    try:
        stock_id = db_manager.get_or_create_stock_id(ticker)
        ld = get_latest_date(ticker)
        
        # Determine start date with proper market close logic
        if ld:
            # Check if we should skip today's data due to market not being closed
            now_market = datetime.now(pytz.timezone("US/Eastern"))
            today_str = now_market.strftime('%Y-%m-%d')
            
            if ld == today_str and now_market.hour < 16:
                # Latest data is today but market hasn't closed - don't fetch new data
                print(f"Skipping {ticker} - latest data is today ({ld}) but market hasn't closed yet")
                return
            else:
                # Safe to fetch from day after latest date
                start_date = (datetime.strptime(ld, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = '2020-01-01'

        print(f"Syncing {ticker} from {start_date}. Latest DB date was {ld}")
        hist = yf_download_with_retry(
            ticker,
            start=start_date,
            progress=False
        )
        if hist.empty:
            print(f"No new data for {ticker}")
        else:
            price_data = []
            for idx, row in hist.iterrows():
                d = idx.strftime('%Y-%m-%d')
                if d < start_date:
                    continue
                if pd.isna(row[['Open', 'High', 'Low', 'Close', 'Volume']]).any():
                    continue
                price_data.append({
                    'date': d,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            if price_data:
                db_manager.store_stock_prices(stock_id, price_data)
                print(f"Stored prices for {ticker} up to {price_data[-1]['date']}")

        industry_data = news_analyzer.get_industry_and_sector(ticker)
        if industry_data:
            db_manager.store_stock_info(stock_id, industry_data['sector'], industry_data['industry'])
        news_analyzer.process_stock_news(ticker, stock_id, analyze_sentiment=analyze_sentiment)

    except Exception as e:
        print(f"Error syncing {ticker}: {e}")

def sync_all_tickers_threaded(all_tickers, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(sync_ticker, ticker, analyze_sentiment=False): ticker for ticker in all_tickers}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Syncing tickers"):
            ticker = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error syncing {ticker}: {e}")

def sync_all_tickers_sequential(all_tickers):
    for ticker in tqdm(all_tickers, desc="Syncing tickers"):
        sync_ticker(ticker)

def sequential_sentiment_pass():
    analyzer = SentimentAnalyzer()
    # Fetch all news items without sentiment_score (or with a null/placeholder value)
    news_items = db_manager.get_news_without_sentiment()
    for news in news_items:
        text = f"{news['title']} {news['summary']}"
        score = analyzer.analyze(text)
        db_manager.update_news_sentiment(news['id'], score)

def get_latest_complete_trading_date(latest_db_date, market_timezone="US/Eastern", market_close_hour=16):
    now_market = datetime.now(pytz.timezone(market_timezone))
    today_str = now_market.strftime('%Y-%m-%d')
    # If latest date is today but market hasn't closed, use previous day
    if latest_db_date == today_str and now_market.hour < market_close_hour:
        # Go back one day (could be weekend/holiday, but yfinance will skip non-trading days)
        prev_day = now_market - timedelta(days=1)
        return prev_day.strftime('%Y-%m-%d')
    return latest_db_date

if __name__ == '__main__':
    initialize_db()
    nasdaq = get_nasdaq_100_tickers()
    tsx    = get_tsx60_tickers()  # TSX-60 tickers parsed dynamically from Wikipedia :contentReference[oaicite:1]{index=1}
    all_tickers = nasdaq + tsx
    sync_all_tickers_threaded(all_tickers)
    sequential_sentiment_pass()
    plot_all(all_tickers)

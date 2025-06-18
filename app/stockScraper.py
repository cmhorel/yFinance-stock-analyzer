# stockScraper.py
import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import app.appconfig as appconfig
import app.news_analyzer as news_analyzer  # NEW: Import news analyzer module
from app.database_manager import DatabaseManager  # NEW: Import centralized database manager

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
            UNIQUE (stock_id, date, title), 
            FOREIGN KEY (stock_id) REFERENCES stocks(id)
        )
    ''')

    conn.commit()
    conn.close()



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
        hist = yf.download(tickers, start=start, group_by='ticker', threads=True, progress=False)
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

def sync_ticker(ticker):
    try:
        # MODIFIED: Use database manager for stock operations
        stock_id = db_manager.get_or_create_stock_id(ticker)

        ld = get_latest_date(ticker)
        start_date = (datetime.strptime(ld, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d') if ld else '2020-01-01'
        start_date = min(start_date, datetime.now().strftime('%Y-%m-%d'))

        hist = yf.download(ticker, start=start_date, progress=False)
        if hist.empty:
            print(f"No data for {ticker}")
        else:
            # Prepare price data for bulk insert
            price_data = []
            for idx, row in hist.iterrows():
                d = idx.strftime('%Y-%m-%d')
                if d < start_date:
                    continue

                # Skip rows with any NaN values in critical columns
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
            
            # MODIFIED: Use database manager to store prices
            if price_data:
                db_manager.store_stock_prices(stock_id, price_data)
        
        # NEW: Fetch and store industry/sector and news after price sync
        industry_data = news_analyzer.get_industry_and_sector(ticker)
        if industry_data:
            db_manager.store_stock_info(stock_id, industry_data['sector'], industry_data['industry'])
        
        news_analyzer.process_stock_news(ticker, stock_id)
        
    except Exception as e:
        print(f"Error syncing {ticker}: {e}")


def sync_all_tickers_sequential(all_tickers):
    for ticker in tqdm(all_tickers, desc="Syncing tickers"):
        sync_ticker(ticker)

if __name__ == '__main__':
    initialize_db()
    nasdaq = get_nasdaq_100_tickers()
    tsx    = get_tsx60_tickers()  # TSX-60 tickers parsed dynamically from Wikipedia :contentReference[oaicite:1]{index=1}
    all_tickers = nasdaq + tsx
    print(all_tickers)
    sync_all_tickers_sequential(all_tickers)
    plot_all(all_tickers)

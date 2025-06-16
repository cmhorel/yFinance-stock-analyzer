# stockSimulator.py
import sqlite3
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import stockAnalyzer  # Import to get recommendations
import config  # For DB_NAME

DB_NAME = config.DB_NAME
PLOTS_DIR = 'plots'
STARTING_CASH = 10000.0
BROKERAGE_FEE = 10.0
MIN_DAYS_BETWEEN_TRADES = 7
MAX_BUY_PER_STOCK = 1000.0  # Max $ to spend per stock for diversification
MAX_SHARES_PER_BUY = 10  # Max shares to buy per stock

def initialize_portfolio_db():
    """Create portfolio-related tables if they don't exist and initialize portfolio."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Portfolio table: Overall state
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cash_balance REAL DEFAULT 0,
            total_value REAL DEFAULT 0,
            last_transaction_date TEXT
        )
    ''')
    
    # Holdings table: Current stock holdings
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER,
            ticker TEXT,
            quantity INTEGER,
            purchase_price REAL,
            purchase_date TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolio(id)
        )
    ''')
    
    # Transactions table: Log of buys/sells
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER,
            ticker TEXT,
            type TEXT,  -- 'buy' or 'sell'
            quantity INTEGER,
            price REAL,
            fee REAL,
            date TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolio(id)
        )
    ''')
    
    # Value history table: For plotting historical portfolio value
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_value_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER,
            date TEXT,
            total_value REAL,
            cash_balance REAL,
            FOREIGN KEY (portfolio_id) REFERENCES portfolio(id)
        )
    ''')
    
    # Initialize portfolio if not exists
    c.execute("SELECT id FROM portfolio")
    if not c.fetchone():
        c.execute("INSERT INTO portfolio (cash_balance, total_value) VALUES (?, ?)", (STARTING_CASH, STARTING_CASH))
        conn.commit()
        print("Initialized portfolio with $10,000 starting cash.")
    
    conn.close()

def get_portfolio_state():
    """Fetch current portfolio state, holdings, and last transaction date."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, cash_balance, last_transaction_date FROM portfolio ORDER BY id DESC LIMIT 1")
    portfolio = c.fetchone()
    if not portfolio:
        conn.close()
        return None, None, None
    
    portfolio_id, cash_balance, last_tx_date = portfolio
    df_holdings = pd.read_sql_query(
        "SELECT ticker, quantity, purchase_price FROM portfolio_holdings WHERE portfolio_id = ?",
        conn, params=(portfolio_id,)
    )
    conn.close()
    return portfolio_id, cash_balance, last_tx_date, df_holdings

def get_current_prices(tickers):
    """Fetch current prices for a list of tickers."""
    prices = {}
    for ticker in tickers:
        try:
            prices[ticker] = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
        except Exception:
            prices[ticker] = None  # Handle errors
    return prices

def generate_portfolio_report(portfolio_id, cash_balance, df_holdings):
    """Generate a report on current portfolio value."""
    if df_holdings.empty:
        print("No holdings. Cash balance: ${:.2f}".format(cash_balance))
        return cash_balance
    
    current_prices = get_current_prices(df_holdings['ticker'].tolist())
    total_value = cash_balance
    print("\nCurrent Portfolio Report:")
    for _, row in df_holdings.iterrows():
        ticker = row['ticker']
        quantity = row['quantity']
        current_price = current_prices.get(ticker, row['purchase_price'])  # Fallback to purchase price
        if current_price is not None:
            holding_value = quantity * current_price
            total_value += holding_value
            print(f"- {ticker}: {quantity} shares @ ${current_price:.2f} (Value: ${holding_value:.2f})")
    
    print(f"Cash Balance: ${cash_balance:.2f}")
    print(f"Total Portfolio Value: ${total_value:.2f}")
    return total_value

def log_portfolio_value(portfolio_id, total_value, cash_balance):
    """Log current portfolio value for historical plotting."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute(
        "INSERT INTO portfolio_value_history (portfolio_id, date, total_value, cash_balance) VALUES (?, ?, ?, ?)",
        (portfolio_id, today, total_value, cash_balance)
    )
    conn.commit()
    conn.close()

def simulate_trades(portfolio_id, cash_balance, df_holdings):
    """Simulate buys and sells based on recommendations."""
    current_holdings = df_holdings['ticker'].tolist() if not df_holdings.empty else []
    recommendations = stockAnalyzer.main(available_cash=cash_balance, current_holdings=current_holdings)
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    new_cash = cash_balance
    
    # Process sells first to free up cash
    for sell in recommendations['sell_candidates']:
        ticker, _, _, _, _, current_price = sell
        if ticker not in current_holdings or current_price is None:
            continue
        holding = df_holdings[df_holdings['ticker'] == ticker].iloc[0]
        quantity = holding['quantity']
        sale_proceeds = quantity * current_price - BROKERAGE_FEE
        new_cash += sale_proceeds
        
        # Log transaction and remove holding
        c.execute(
            "INSERT INTO portfolio_transactions (portfolio_id, ticker, type, quantity, price, fee, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (portfolio_id, ticker, 'sell', quantity, current_price, BROKERAGE_FEE, today)
        )
        c.execute("DELETE FROM portfolio_holdings WHERE portfolio_id = ? AND ticker = ?", (portfolio_id, ticker))
        print(f"Sold {quantity} shares of {ticker} @ ${current_price:.2f} (Proceeds: ${sale_proceeds:.2f} after fee)")
    
    # Process buys with updated cash
    for buy in recommendations['buy_candidates']:
        ticker, _, _, _, _, current_price = buy
        if current_price is None or new_cash < current_price + BROKERAGE_FEE:
            continue
        # Calculate affordable quantity (up to max shares or $1000)
        max_affordable = min(MAX_SHARES_PER_BUY, int(min(new_cash - BROKERAGE_FEE, MAX_BUY_PER_STOCK) / current_price))
        if max_affordable < 1:
            continue
        cost = max_affordable * current_price + BROKERAGE_FEE
        new_cash -= cost
        
        # Log transaction and add holding
        c.execute(
            "INSERT INTO portfolio_transactions (portfolio_id, ticker, type, quantity, price, fee, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (portfolio_id, ticker, 'buy', max_affordable, current_price, BROKERAGE_FEE, today)
        )
        c.execute(
            "INSERT INTO portfolio_holdings (portfolio_id, ticker, quantity, purchase_price, purchase_date) VALUES (?, ?, ?, ?, ?)",
            (portfolio_id, ticker, max_affordable, current_price, today)
        )
        print(f"Bought {max_affordable} shares of {ticker} @ ${current_price:.2f} (Cost: ${cost:.2f} including fee)")
    
    # Update portfolio
    c.execute(
        "UPDATE portfolio SET cash_balance = ?, last_transaction_date = ? WHERE id = ?",
        (new_cash, today, portfolio_id)
    )
    conn.commit()
    conn.close()
    return new_cash

def create_portfolio_value_plot(portfolio_id):
    """Create and save a stacked line graph of portfolio value and holdings."""
    conn = sqlite3.connect(DB_NAME)
    df_history = pd.read_sql_query(
        "SELECT date, total_value, cash_balance FROM portfolio_value_history WHERE portfolio_id = ? ORDER BY date",
        conn, params=(portfolio_id,)
    )
    df_holdings_history = pd.read_sql_query(
        """
        SELECT ph.purchase_date AS date, ph.ticker, ph.quantity, sp.close AS price
        FROM portfolio_holdings ph
        LEFT JOIN stock_prices sp ON ph.ticker = sp.symbol AND ph.purchase_date <= sp.date
        WHERE ph.portfolio_id = ?
        """, conn, params=(portfolio_id,)
    )
    conn.close()
    
    if df_history.empty:
        print("No historical data for plotting.")
        return
    
    df_history['date'] = pd.to_datetime(df_history['date'])
    fig = make_subplots(rows=1, cols=1)
    
    # Total portfolio value
    fig.add_trace(go.Scatter(x=df_history['date'], y=df_history['total_value'], mode='lines', name='Total Value'))
    
    # Stacked holdings value (simplified: aggregate by ticker over time)
    if not df_holdings_history.empty:
        df_holdings_history['date'] = pd.to_datetime(df_holdings_history['date'])
        for ticker in df_holdings_history['ticker'].unique():
            df_ticker = df_holdings_history[df_holdings_history['ticker'] == ticker]
            fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker['quantity'] * df_ticker['price'], mode='lines', name=ticker, stackgroup='holdings'))
    
    fig.update_layout(title='Portfolio Value and Holdings Over Time', xaxis_title='Date', yaxis_title='Value ($)', height=600)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    filename = os.path.join(PLOTS_DIR, "portfolio_value.html")
    fig.write_html(filename)
    print(f"Saved portfolio value plot to {filename}")

def main():
    initialize_portfolio_db()
    portfolio_id, cash_balance, last_tx_date, df_holdings = get_portfolio_state()
    if portfolio_id is None:
        print("Error: Portfolio not initialized.")
        return
    
    today = datetime.now()
    if last_tx_date:
        last_tx = datetime.strptime(last_tx_date, '%Y-%m-%d')
        days_since = (today - last_tx).days
        if days_since < MIN_DAYS_BETWEEN_TRADES:
            print(f"Too soon! Only {days_since} days since last transaction. No trades performed.")
            total_value = generate_portfolio_report(portfolio_id, cash_balance, df_holdings)
            log_portfolio_value(portfolio_id, total_value, cash_balance)
            create_portfolio_value_plot(portfolio_id)
            return
    
    # Perform trades if eligible
    print("Eligible for trades. Running analysis...")
    new_cash = simulate_trades(portfolio_id, cash_balance, df_holdings)
    # Reload holdings after trades
    _, _, _, df_holdings = get_portfolio_state()
    total_value = generate_portfolio_report(portfolio_id, new_cash, df_holdings)
    log_portfolio_value(portfolio_id, total_value, new_cash)
    create_portfolio_value_plot(portfolio_id)

if __name__ == "__main__":
    main()
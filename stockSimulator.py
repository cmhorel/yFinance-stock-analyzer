# stockSimulator.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf
import stockAnalyzer
import config
import os

def initialize_portfolio_db():
    """Initialize the portfolio simulation database tables."""
    conn = sqlite3.connect(config.DB_NAME)
    c = conn.cursor()
    
    # Create portfolio_state table to track overall portfolio
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY,
            cash_balance REAL,
            total_portfolio_value REAL,
            last_transaction_date TEXT,
            created_date TEXT
        )
    ''')
    
    # Create portfolio_holdings table to track individual stock positions
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_holdings (
            stock_id INTEGER,
            symbol TEXT,
            quantity INTEGER,
            avg_cost_per_share REAL,
            total_cost REAL,
            PRIMARY KEY (stock_id),
            FOREIGN KEY (stock_id) REFERENCES stocks(id)
        )
    ''')
    
    # Create portfolio_transactions table to track all buy/sell transactions
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER,
            symbol TEXT,
            transaction_type TEXT,  -- 'BUY' or 'SELL'
            quantity INTEGER,
            price_per_share REAL,
            total_amount REAL,
            brokerage_fee REAL,
            transaction_date TEXT,
            FOREIGN KEY (stock_id) REFERENCES stocks(id)
        )
    ''')
    
    # Initialize portfolio with $10,000 if it doesn't exist
    c.execute('SELECT COUNT(*) FROM portfolio_state')
    if c.fetchone()[0] == 0:
        c.execute('''
            INSERT INTO portfolio_state (cash_balance, total_portfolio_value, last_transaction_date, created_date)
            VALUES (?, ?, ?, ?)
        ''', (10000.0, 10000.0, None, datetime.now().strftime('%Y-%m-%d')))
        print("Initialized new portfolio with $10,000")
    
    conn.commit()
    conn.close()

def get_portfolio_state():
    """Get current portfolio state."""
    conn = sqlite3.connect(config.DB_NAME)
    c = conn.cursor()
    
    c.execute('SELECT * FROM portfolio_state ORDER BY id DESC LIMIT 1')
    portfolio = c.fetchone()
    
    c.execute('''
        SELECT ph.*, s.symbol 
        FROM portfolio_holdings ph
        JOIN stocks s ON ph.stock_id = s.id
        WHERE ph.quantity > 0
    ''')
    holdings = c.fetchall()
    
    conn.close()
    return portfolio, holdings

def get_current_stock_prices(symbols):
    """Get current stock prices for given symbols."""
    if not symbols:
        return {}
    
    try:
        tickers = yf.download(symbols, period='1d', progress=False)
        if len(symbols) == 1:
            return {symbols[0]: tickers['Close'].iloc[-1]}
        else:
            prices = {}
            for symbol in symbols:
                if symbol in tickers['Close'].columns:
                    prices[symbol] = tickers['Close'][symbol].iloc[-1]
            return prices
    except Exception as e:
        print(f"Error fetching current prices: {e}")
        return {}

def calculate_current_portfolio_value():
    """Calculate current total portfolio value including cash and holdings."""
    portfolio, holdings = get_portfolio_state()
    if not portfolio:
        return 0.0
    
    cash_balance = portfolio[1]
    
    if not holdings:
        return cash_balance
    
    # Get current prices for all held stocks
    symbols = [holding[1] for holding in holdings]  # symbol is at index 6
    current_prices = get_current_stock_prices(symbols)
    
    holdings_value = 0.0
    for holding in holdings:
        symbol = holding[1]
        quantity = holding[2]
        if symbol in current_prices:
            holdings_value += quantity * current_prices[symbol]
    
    return cash_balance + holdings_value

def can_make_transactions():
    """Check if enough time has passed since last transaction."""
    portfolio, _ = get_portfolio_state()
    if not portfolio or not portfolio[3]:  # No last transaction date
        return True
    
    last_transaction = datetime.strptime(portfolio[3], '%Y-%m-%d')
    days_since = (datetime.now() - last_transaction).days
    
    return days_since >= 7

def get_portfolio_report():
    """Generate a brief portfolio report."""
    portfolio, holdings = get_portfolio_state()
    if not portfolio:
        return "No portfolio found."
    
    cash_balance = portfolio[1]
    current_total_value = calculate_current_portfolio_value()
    
    report = f"\n=== PORTFOLIO REPORT ===\n"
    report += f"Cash Balance: ${cash_balance:,.2f}\n"
    report += f"Current Total Portfolio Value: ${current_total_value:,.2f}\n"
    report += f"Total Return: ${current_total_value - 10000:,.2f} ({((current_total_value/10000 - 1) * 100):+.2f}%)\n"
    
    if holdings:
        report += f"\n=== CURRENT HOLDINGS ===\n"
        symbols = [holding[1] for holding in holdings]
        current_prices = get_current_stock_prices(symbols)
        
        for holding in holdings:
            symbol = holding[1]
            quantity = holding[2]
            avg_cost = holding[3]
            total_cost = holding[4]
            
            if symbol in current_prices:
                current_price = current_prices[symbol]
                current_value = quantity * current_price
                gain_loss = current_value - total_cost
                gain_loss_pct = (gain_loss / total_cost) * 100 if total_cost > 0 else 0
                
                report += f"{symbol}: {quantity} shares @ ${avg_cost:.2f} avg cost\n"
                report += f"  Current: ${current_price:.2f}/share, Total: ${current_value:,.2f}\n"
                report += f"  Gain/Loss: ${gain_loss:+,.2f} ({gain_loss_pct:+.2f}%)\n"
    
    return report

def execute_buy_transaction(stock_id, symbol, quantity, price_per_share):
    """Execute a buy transaction."""
    conn = sqlite3.connect(config.DB_NAME)
    c = conn.cursor()
    
    brokerage_fee = 10.0
    total_cost = (quantity * price_per_share) + brokerage_fee
    
    # Record transaction
    c.execute('''
        INSERT INTO portfolio_transactions 
        (stock_id, symbol, transaction_type, quantity, price_per_share, total_amount, brokerage_fee, transaction_date)
        VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?)
    ''', (stock_id, symbol, quantity, price_per_share, total_cost, brokerage_fee, datetime.now().strftime('%Y-%m-%d')))
    
    # Update or insert holding
    c.execute('SELECT * FROM portfolio_holdings WHERE stock_id = ?', (stock_id,))
    existing_holding = c.fetchone()
    
    if existing_holding:
        # Update existing holding
        old_quantity = existing_holding[2]
        old_total_cost = existing_holding[4]
        new_quantity = old_quantity + quantity
        new_total_cost = old_total_cost + total_cost
        new_avg_cost = new_total_cost / new_quantity
        
        c.execute('''
            UPDATE portfolio_holdings 
            SET quantity = ?, avg_cost_per_share = ?, total_cost = ?
            WHERE stock_id = ?
        ''', (new_quantity, new_avg_cost, new_total_cost, stock_id))
    else:
        # Insert new holding
        c.execute('''
            INSERT INTO portfolio_holdings (stock_id, symbol, quantity, avg_cost_per_share, total_cost)
            VALUES (?, ?, ?, ?, ?)
        ''', (stock_id, symbol, quantity, price_per_share, total_cost))
    
    # Update portfolio cash balance
    c.execute('SELECT cash_balance FROM portfolio_state ORDER BY id DESC LIMIT 1')
    current_cash = c.fetchone()[0]
    new_cash = current_cash - total_cost
    
    c.execute('''
        UPDATE portfolio_state 
        SET cash_balance = ?, last_transaction_date = ?
        WHERE id = (SELECT MAX(id) FROM portfolio_state)
    ''', (new_cash, datetime.now().strftime('%Y-%m-%d')))
    
    conn.commit()
    conn.close()
    
    print(f"BUY: {quantity} shares of {symbol} @ ${price_per_share:.2f} (Total: ${total_cost:.2f})")

def execute_sell_transaction(stock_id, symbol, quantity, price_per_share):
    """Execute a sell transaction."""
    conn = sqlite3.connect(config.DB_NAME)
    c = conn.cursor()
    
    brokerage_fee = 10.0
    total_proceeds = (quantity * price_per_share) - brokerage_fee
    
    # Record transaction
    c.execute('''
        INSERT INTO portfolio_transactions 
        (stock_id, symbol, transaction_type, quantity, price_per_share, total_amount, brokerage_fee, transaction_date)
        VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?)
    ''', (stock_id, symbol, quantity, price_per_share, total_proceeds, brokerage_fee, datetime.now().strftime('%Y-%m-%d')))
    
    # Update holding
    c.execute('SELECT * FROM portfolio_holdings WHERE stock_id = ?', (stock_id,))
    holding = c.fetchone()
    
    if holding:
        old_quantity = holding[2]
        old_total_cost = holding[4]
        new_quantity = old_quantity - quantity
        
        if new_quantity <= 0:
            # Remove holding entirely
            c.execute('DELETE FROM portfolio_holdings WHERE stock_id = ?', (stock_id,))
        else:
            # Reduce holding proportionally
            cost_per_share = old_total_cost / old_quantity
            new_total_cost = new_quantity * cost_per_share
            c.execute('''
                UPDATE portfolio_holdings 
                SET quantity = ?, total_cost = ?
                WHERE stock_id = ?
            ''', (new_quantity, new_total_cost, stock_id))
    
    # Update portfolio cash balance
    c.execute('SELECT cash_balance FROM portfolio_state ORDER BY id DESC LIMIT 1')
    current_cash = c.fetchone()[0]
    new_cash = current_cash + total_proceeds
    
    c.execute('''
        UPDATE portfolio_state 
        SET cash_balance = ?, last_transaction_date = ?
        WHERE id = (SELECT MAX(id) FROM portfolio_state)
    ''', (new_cash, datetime.now().strftime('%Y-%m-%d')))
    
    conn.commit()
    conn.close()
    
    print(f"SELL: {quantity} shares of {symbol} @ ${price_per_share:.2f} (Proceeds: ${total_proceeds:.2f})")

def get_affordable_recommendations(buy_candidates, cash_available):
    """Filter buy recommendations to only include affordable stocks."""
    if not buy_candidates:
        return []
    
    affordable = []
    symbols = [candidate[0] for candidate in buy_candidates]
    current_prices = get_current_stock_prices(symbols)
    
    for candidate in buy_candidates:
        symbol = candidate[0]
        if symbol in current_prices:
            price = current_prices[symbol]
            # Account for brokerage fee - need at least price + $10
            if cash_available >= (price + 10):
                affordable.append(candidate + (price,))  # Add current price to tuple
    
    return affordable

def execute_portfolio_transactions():
    """Execute buy/sell transactions based on stock analysis."""
    # Get current portfolio state
    portfolio, holdings = get_portfolio_state()
    cash_balance = portfolio[1]
    
    # Get stock recommendations
    conn = sqlite3.connect(config.DB_NAME)
    df = stockAnalyzer.get_stock_data(conn)
    conn.close()
    
    if df.empty:
        print("No stock data available for analysis.")
        return
    
    # Run analysis to get recommendations
    buy_candidates = []
    sell_candidates = []
    
    grouped = df.groupby('symbol')
    for ticker, group in grouped:
        result = stockAnalyzer.analyze_ticker(group, df)
        if result is None:
            continue
        
        if result['buy_score'] >= 5:
            buy_candidates.append((ticker, result['buy_score'], result['avg_sentiment'], result['industry'], result['sector']))
        if result['sell_score'] >= 5:
            sell_candidates.append((ticker, result['sell_score'], result['avg_sentiment'], result['industry'], result['sector']))
    
    # Sort by score
    buy_candidates.sort(key=lambda x: x[1], reverse=True)
    sell_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Execute sell transactions first
    held_symbols = {holding[6]: holding for holding in holdings}  # symbol -> holding data
    
    for sell_candidate in sell_candidates[:5]:  # Limit to top 5 sell candidates
        symbol = sell_candidate[0]
        if symbol in held_symbols:
            holding = held_symbols[symbol]
            stock_id = holding[0]
            quantity = holding[2]
            
            # Get current price
            current_prices = get_current_stock_prices([symbol])
            if symbol in current_prices:
                current_price = current_prices[symbol]
                execute_sell_transaction(stock_id, symbol, quantity, current_price)
                cash_balance += (quantity * current_price) - 10  # Update available cash
    
    # Execute buy transactions
    affordable_buys = get_affordable_recommendations(buy_candidates[:10], cash_balance)
    
    # Get stock IDs for buy candidates
    conn = sqlite3.connect(config.DB_NAME)
    c = conn.cursor()
    
    for buy_candidate in affordable_buys[:5]:  # Limit to top 5 affordable buys
        symbol = buy_candidate[0]
        current_price = buy_candidate[-1]  # Price was added by get_affordable_recommendations
        
        # Calculate how many shares we can afford
        max_affordable = int((cash_balance - 10) / current_price)  # Account for brokerage fee
        
        if max_affordable > 0:
            # Buy a reasonable amount (limit to $1000 per position or 10% of cash, whichever is smaller)
            max_position_value = min(1000, cash_balance * 0.1)
            target_quantity = min(max_affordable, int(max_position_value / current_price))
            
            if target_quantity > 0:
                # Get stock ID
                c.execute('SELECT id FROM stocks WHERE symbol = ?', (symbol,))
                stock_id_row = c.fetchone()
                if stock_id_row:
                    stock_id = stock_id_row[0]
                    execute_buy_transaction(stock_id, symbol, target_quantity, current_price)
                    cash_balance -= (target_quantity * current_price) + 10  # Update available cash
    
    conn.close()
    
    # Update total portfolio value
    new_total_value = calculate_current_portfolio_value()
    conn = sqlite3.connect(config.DB_NAME)
    c = conn.cursor()
    c.execute('''
        UPDATE portfolio_state 
        SET total_portfolio_value = ?
        WHERE id = (SELECT MAX(id) FROM portfolio_state)
    ''', (new_total_value,))
    conn.commit()
    conn.close()

def create_portfolio_performance_plot(save_path='plots'):
    """Create portfolio performance visualization using actual DB state."""
    os.makedirs(save_path, exist_ok=True)
    conn = sqlite3.connect(config.DB_NAME)

    # Get all transaction dates
    transactions_df = pd.read_sql_query(
        'SELECT transaction_date FROM portfolio_transactions ORDER BY transaction_date ASC', conn)
    if transactions_df.empty:
        print("No transactions found for portfolio visualization.")
        return

    # Create date range from first transaction to today
    start_date = pd.to_datetime(transactions_df['transaction_date'].min())
    end_date = pd.Timestamp.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    portfolio_values, cash_values, holdings_values = [], [], []

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')

        # Get latest cash balance as of this date
        cash_query = '''
            SELECT cash_balance FROM portfolio_state
            WHERE last_transaction_date <= ?
            ORDER BY last_transaction_date DESC LIMIT 1
        '''
        cur = conn.execute(cash_query, (date_str,))
        row = cur.fetchone()
        cash = row[0] if row else 10000.0  # fallback to initial cash

        # Get current holdings as of this date
        holdings_query = '''
            SELECT ph.symbol, ph.quantity
            FROM portfolio_holdings ph
            JOIN stocks s ON ph.stock_id = s.id
            WHERE ph.quantity > 0
        '''
        holdings = pd.read_sql_query(holdings_query, conn)
        holdings_value = 0.0
        if not holdings.empty:
            symbols = holdings['symbol'].tolist()
            try:
                # Fetch a window of 5 days up to the current date
                prices = yf.download(
                    symbols, 
                    start=(date - timedelta(days=5)).strftime('%Y-%m-%d'), 
                    end=(date + timedelta(days=1)).strftime('%Y-%m-%d'), 
                    progress=False
                )
                for idx, row in holdings.iterrows():
                    symbol = row['symbol']
                    qty = row['quantity']
                    price = 0
                    # Try to get the last available close price
                    if len(symbols) == 1:
                        if not prices.empty and 'Close' in prices.columns:
                            price = prices['Close'].dropna().iloc[-1]
                    else:
                        if 'Close' in prices and symbol in prices['Close']:
                            price_series = prices['Close'][symbol].dropna()
                            if not price_series.empty:
                                price = price_series.iloc[-1]
                    holdings_value += qty * price
            except Exception as e:
                print(f"Price fetch failed for {symbols} on {date_str}: {e}")
                holdings_value = holdings_values[-1] if holdings_values else 0.0
        cash_values.append(cash)
        print(cash_values)
        holdings_values.append(holdings_value)
        portfolio_values.append(cash + holdings_value)

    conn.close()

    # ...plotting code unchanged...
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Portfolio Value Over Time', 'Portfolio Composition'),
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=portfolio_values,
            mode='lines+markers',
            name='Total Portfolio Value',
            marker=dict(size=20, color='blue', symbol='circle'),
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Portfolio Value: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_hline(y=10000, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=cash_values,
            mode='lines+markers',
            marker=dict(size=20, color='green', symbol='circle'),
            name='Cash',
            fill='tonexty',
            line=dict(color='green'),
            hovertemplate='Date: %{x}<br>Cash: $%{y:,.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=holdings_values,#[cash + holding for cash, holding in zip(cash_values, holdings_values)],
            mode='lines+markers',
            marker=dict(size=20, color='orange', symbol='circle'),
            name='Holdings',
            fill='tonexty',
            line=dict(color='orange'),
            hovertemplate='Date: %{x}<br>Holdings Value: $%{y:,.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    current_value = portfolio_values[-1]
    total_return = current_value - 10000
    return_pct = (total_return / 10000) * 100

    fig.update_layout(
        title=f'Portfolio Performance - Total Return: ${total_return:+,.2f} ({return_pct:+.2f}%)',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    filename = os.path.join(save_path, "portfolio_performance.html")
    fig.write_html(filename)
    print(f"Saved portfolio performance plot to {filename}")

    return fig

def main():
    """Main function to run the portfolio simulator."""
    print("=== STOCK PORTFOLIO SIMULATOR ===")
    
    # Initialize database
    initialize_portfolio_db()
    
    # Check if we can make transactions
    if not can_make_transactions():
        print("⚠️  Less than a week has passed since last transaction. No new trades will be executed.")
        print(get_portfolio_report())
    else:
        print("✅ Ready to execute new transactions based on stock analysis.")
        print("Current portfolio state:")
        print(get_portfolio_report())
        
        print("\n🔄 Executing portfolio transactions...")
        execute_portfolio_transactions()
        
        print("\n📊 Updated portfolio state:")
        print(get_portfolio_report())
    
    # Create portfolio performance visualization
    print("\n📈 Creating portfolio performance visualization...")
    create_portfolio_performance_plot()
    
    print("\n✅ Portfolio simulation complete!")

if __name__ == "__main__":
    main()
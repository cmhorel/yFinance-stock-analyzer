# stockSimulator.py
from app.database_manager import db_manager  # NEW: Import centralized database manager
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import app.appconfig as appconfig
import app.stockAnalyzer as stockAnalyzer
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import sqlite3
import yfinance as yf


TIME_ZONE  = pytz.timezone(appconfig.TIME_ZONE)  # NEW: Use timezone from appconfig


def initialize_portfolio_db():
    """Initialize the portfolio simulation database tables."""
    conn = sqlite3.connect(appconfig.DB_NAME)
    c = conn.cursor()
    
    # Create portfolio_state table to track overall portfolio
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cash_balance REAL,
            total_portfolio_value REAL,
            last_transaction_date TEXT,
            created_date TEXT UNIQUE
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
    
    today = datetime.now(TIME_ZONE).strftime('%Y-%m-%d')
    c.execute('SELECT COUNT(*) FROM portfolio_state')
    if c.fetchone()[0] == 0:
        c.execute('''
            INSERT INTO portfolio_state (cash_balance, total_portfolio_value, last_transaction_date, created_date)
            VALUES (?, ?, ?, ?)
        ''', (10000.0, 10000.0, None, today))
        print("Initialized new portfolio with $10,000")
    
    conn.commit()
    conn.close()

def get_portfolio_state():
    """Get current portfolio state using database manager."""
    return db_manager.get_portfolio_state()

def get_current_stock_prices(symbols):
    """Get current stock prices for given symbols."""
    if not symbols:
        return {}
    
    try:
        tickers = yf.download(symbols, period='1d', progress=False, auto_adjust=False)
        if len(symbols) == 1:
            return {symbols[0]: tickers['Close'][symbols[0]].iloc[-1]}
        else:
            prices = {}
            for symbol in symbols:
                series = tickers['Close'][symbol].dropna()
                if not series.empty:
                    prices[symbol] = series.iloc[-1]
                    
            return prices
    except Exception as e:
        print(f"Error fetching current prices: {e}")
        return {}

def calculate_current_portfolio_value_with_cash(cash_balance):
    """Calculate portfolio value with given cash balance."""
    _, holdings = get_portfolio_state()
    
    if not holdings:
        return cash_balance
    
    # Get current prices for all held stocks
    symbols = [holding[1] for holding in holdings]
    current_prices = get_current_stock_prices(symbols)
    
    holdings_value = 0.0
    for holding in holdings:
        symbol = holding[1]
        quantity = holding[2]
        if symbol in current_prices:
            holdings_value += quantity * current_prices[symbol]
    
    return cash_balance + holdings_value

def calculate_current_portfolio_value():
    """Calculate current total portfolio value including cash and holdings."""
    portfolio, holdings = get_portfolio_state()
    if not portfolio:
        return 0.0
    
    cash_balance = portfolio[1]
    
    if not holdings:
        return cash_balance
    
    # Use database prices for consistency (same as reconstruct_holdings_and_value)
    holdings_value = 0.0
    conn = sqlite3.connect(appconfig.DB_NAME)
    
    for holding in holdings:
        symbol = holding[1]
        quantity = holding[2]
        
        # Get latest price from database
        cursor = conn.execute('''
            SELECT sp.close FROM stock_prices sp
            JOIN stocks s ON sp.stock_id = s.id
            WHERE s.symbol = ? 
            ORDER BY sp.date DESC LIMIT 1
        ''', (symbol,))
        result = cursor.fetchone()
        
        if result:
            price = result[0]
            holdings_value += quantity * price
        else:
            print(f"Warning: No price data for {symbol} in database")
    
    conn.close()
    return cash_balance + holdings_value

def can_make_transactions():
    """Check if enough time has passed since last transaction."""
    portfolio, _ = get_portfolio_state()
    if not portfolio or not portfolio[3]:  # No last transaction date
        return True
    
    last_transaction = datetime.strptime(portfolio[3], '%Y-%m-%d')
    days_since = (datetime.now(TIME_ZONE) - last_transaction.replace(tzinfo=TIME_ZONE)).days

    return days_since >= 1

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
        print(current_prices)
        
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
    """Execute a buy transaction using database manager."""
    brokerage_fee = 10.0
    total_cost = (quantity * price_per_share) + brokerage_fee
    
    # Record transaction
    db_manager.record_transaction(
        stock_id, symbol, 'BUY', quantity, price_per_share, 
        total_cost, brokerage_fee, datetime.now(TIME_ZONE).strftime('%Y-%m-%d')
    )
    
    # Update or insert holding
    db_manager.update_or_create_holding(stock_id, symbol, quantity, price_per_share, total_cost - brokerage_fee)
    
    # Get current portfolio state and create new entry
    portfolio, _ = get_portfolio_state()
    current_cash = portfolio[1]
    new_cash = current_cash - total_cost
    transaction_date = datetime.now(TIME_ZONE).strftime('%Y-%m-%d')
    
    conn = sqlite3.connect(appconfig.DB_NAME)
    c = conn.cursor()
    # Check if an entry exists for today
    c.execute('SELECT id FROM portfolio_state WHERE created_date = ?', (transaction_date,))
    row = c.fetchone()
    if row:
        # Update existing entry for today
        c.execute('''
            UPDATE portfolio_state
            SET cash_balance = ?, total_portfolio_value = ?, last_transaction_date = ?
            WHERE id = ?
        ''', (new_cash, calculate_current_portfolio_value_with_cash(new_cash), transaction_date, row[0]))
    else:
        # Insert new entry for today
        c.execute('''
            INSERT INTO portfolio_state (cash_balance, total_portfolio_value, last_transaction_date, created_date)
            VALUES (?, ?, ?, ?)
        ''', (new_cash, calculate_current_portfolio_value_with_cash(new_cash), transaction_date, transaction_date))
    conn.commit()
    conn.close()
    
    print(f"BUY: {quantity} shares of {symbol} @ ${price_per_share:.2f} (Total: ${total_cost:.2f})")

def execute_sell_transaction(stock_id, symbol, quantity, price_per_share):
    """Execute a sell transaction using database manager."""
    brokerage_fee = 10.0
    total_proceeds = (quantity * price_per_share) - brokerage_fee
    
    # Record transaction
    db_manager.record_transaction(
        stock_id, symbol, 'SELL', quantity, price_per_share, 
        total_proceeds, brokerage_fee, datetime.now(TIME_ZONE).strftime('%Y-%m-%d')
    )
    
    # Update holding
    db_manager.reduce_or_remove_holding(stock_id, quantity)
    
    # Get current portfolio state and create new entry
    portfolio, _ = get_portfolio_state()
    current_cash = portfolio[1]
    new_cash = current_cash + total_proceeds
    transaction_date = datetime.now(TIME_ZONE).strftime('%Y-%m-%d')
    
    conn = sqlite3.connect(appconfig.DB_NAME)
    c = conn.cursor()
    # Check if an entry exists for today
    c.execute('SELECT id FROM portfolio_state WHERE created_date = ?', (transaction_date,))
    row = c.fetchone()
    if row:
        # Update existing entry for today
        c.execute('''
            UPDATE portfolio_state
            SET cash_balance = ?, total_portfolio_value = ?, created_date = ?
            WHERE id = ?
        ''', (new_cash, calculate_current_portfolio_value_with_cash(new_cash), transaction_date, row[0]))
    else:
        # Insert new entry for today
        c.execute('''
            INSERT INTO portfolio_state (cash_balance, total_portfolio_value, last_transaction_date, created_date)
            VALUES (?, ?, ?, ?)
        ''', (new_cash, calculate_current_portfolio_value_with_cash(new_cash), transaction_date, transaction_date))
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
    
    # Create a map of currently held stocks for easy lookup
    held_stocks = {}
    for holding in holdings:
        symbol = holding[1]  # symbol is at index 1
        held_stocks[symbol] = {
            'stock_id': holding[0],
            'quantity': holding[2],
            'avg_cost': holding[3],
            'total_cost': holding[4]
        }
    
    # MODIFIED: Use database manager for stock data
    df = db_manager.get_stock_data()
    
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
        
        # Only consider selling stocks we actually hold
        if result['sell_score'] >= 5 and ticker in held_stocks:
            sell_candidates.append((ticker, result['sell_score'], result['avg_sentiment'], result['industry'], result['sector']))
        
        # Consider buying stocks (including adding to existing positions)
        if result['buy_score'] >= 5:
            buy_candidates.append((ticker, result['buy_score'], result['avg_sentiment'], result['industry'], result['sector']))
    
    # Sort by score
    buy_candidates.sort(key=lambda x: x[1], reverse=True)
    sell_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Execute sell transactions first
    for sell_candidate in sell_candidates[:5]:  # Limit to top 5 sell candidates
        symbol = sell_candidate[0]
        if symbol in held_stocks:
            holding_info = held_stocks[symbol]
            stock_id = holding_info['stock_id']
            quantity = holding_info['quantity']
            
            # Get current price
            current_prices = get_current_stock_prices([symbol])
            if symbol in current_prices:
                current_price = current_prices[symbol]
                execute_sell_transaction(stock_id, symbol, quantity, current_price)
                cash_balance += (quantity * current_price) - 10  # Update available cash
                # Remove from held_stocks since we sold everything
                del held_stocks[symbol]
    
    # Execute buy transactions
    affordable_buys = get_affordable_recommendations(buy_candidates[:10], cash_balance)
    
    for buy_candidate in affordable_buys[:5]:
        symbol = buy_candidate[0]
        current_price = buy_candidate[-1]
        
        # Calculate position sizing
        max_affordable = int((cash_balance - 10) / current_price)
        
        if max_affordable > 0:
            # Limit position size to $1000 or 10% of portfolio, whichever is smaller
            max_position_value = min(1000, cash_balance * 0.1)
            target_quantity = min(max_affordable, int(max_position_value / current_price))
            
            # If we already hold this stock, consider current position size
            if symbol in held_stocks:
                current_position_value = held_stocks[symbol]['quantity'] * current_price
                remaining_budget = max_position_value - current_position_value
                if remaining_budget > 0:
                    target_quantity = min(target_quantity, int(remaining_budget / current_price))
            
            if target_quantity > 0:
                stock_id = db_manager.get_or_create_stock_id(symbol)
                execute_buy_transaction(stock_id, symbol, target_quantity, current_price)
                cash_balance -= (target_quantity * current_price) + 10

def reconstruct_holdings_and_value(target_date, transactions_df):
    """Reconstruct holdings and their value as of target_date."""
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    date_str = target_date.strftime('%Y-%m-%d')
    
    # Get all transactions up to and including this date
    day_transactions = transactions_df[transactions_df['transaction_date'] <= date_str]
    holdings = {}
    for _, tx in day_transactions.iterrows():
        symbol = tx['symbol']
        qty = tx['quantity'] if tx['transaction_type'] == 'BUY' else -tx['quantity']
        holdings[symbol] = holdings.get(symbol, 0) + qty
        if holdings[symbol] <= 0:
            holdings.pop(symbol, None)
    
    # Calculate holdings value using database prices as fallback
    holdings_value = 0.0
    if holdings:
        # Try to get prices from database first (more reliable)
        conn = sqlite3.connect(appconfig.DB_NAME)
        for symbol, qty in holdings.items():
            price = 0.0
            try:
                # Get the closest price from database
                cursor = conn.execute('''
                    SELECT sp.close FROM stock_prices sp
                    JOIN stocks s ON sp.stock_id = s.id
                    WHERE s.symbol = ? AND sp.date <= ?
                    ORDER BY sp.date DESC LIMIT 1
                ''', (symbol, date_str))
                result = cursor.fetchone()
                if result:
                    price = result[0]
                else:
                    print(f"Warning: No price data found for {symbol} on or before {date_str}")
            except Exception as e:
                print(f"Database price lookup failed for {symbol}: {e}")
            
            holdings_value += qty * price
        conn.close()
    
    return holdings, holdings_value

def create_portfolio_performance_plot(save_path=appconfig.PLOTS_PATH):
    """Create portfolio performance visualization using database manager."""
    os.makedirs(save_path, exist_ok=True)
    transactions_df = db_manager.get_transactions_df()
    # Get all portfolio states ordered by date
    conn = sqlite3.connect(appconfig.DB_NAME)
    portfolio_states_df = pd.read_sql_query('''
        SELECT * FROM portfolio_state 
        ORDER BY created_date ASC
    ''', conn)
    conn.close()
    
    if portfolio_states_df.empty:
        print("No portfolio states found for visualization.")
        return

    # Convert dates and get unique dates
    portfolio_states_df['created_date'] = pd.to_datetime(portfolio_states_df['created_date'])
    date_range = portfolio_states_df['created_date'].tolist()

    portfolio_values = []
    cash_values = []
    holdings_values = []

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        # Get portfolio state for this specific date
        state_row = portfolio_states_df[portfolio_states_df['created_date'] == date].iloc[0]
        cash = state_row['cash_balance']
        # Calculate holdings value for this date
        holdings, holdings_value = reconstruct_holdings_and_value(date, transactions_df)
        cash_values.append(cash)
        holdings_values.append(holdings_value)
        portfolio_values.append(cash + holdings_value)

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
            y=holdings_values,
            mode='lines+markers',
            marker=dict(size=20, color='orange', symbol='circle'),
            name='Holdings',
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

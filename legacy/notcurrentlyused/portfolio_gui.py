"""
Portfolio Management GUI Application

This application provides a comprehensive GUI for managing stock portfolios with features from stockSimulator.py:
- View portfolio state and holdings
- Execute buy/sell transactions
- Automatic portfolio updates based on time intervals
- Portfolio performance visualization
- Real-time stock data integration
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf
import sqlite3
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import our modules
import app.appconfig as appconfig
import app.stockSimulator as stockSimulator
import app.stockAnalyzer as stockAnalyzer
from app.database_manager import db_manager
import pytz

TIME_ZONE = pytz.timezone(appconfig.TIME_ZONE)


class PortfolioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Portfolio Manager")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize database
        stockSimulator.initialize_portfolio_db()
        
        # Track last update time for automatic portfolio updates
        self.last_update_check = datetime.now(TIME_ZONE)
        
        # Create main interface
        self.create_widgets()
        
        # Start background update checker
        self.start_background_updates()
        
        # Initial data load
        self.refresh_portfolio_data()
    
    def create_widgets(self):
        """Create the main GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Portfolio Overview Tab
        self.create_portfolio_tab()
        
        # Trading Tab
        self.create_trading_tab()
        
        # Holdings Tab
        self.create_holdings_tab()
        
        # Performance Tab
        self.create_performance_tab()
        
        # Settings Tab
        self.create_settings_tab()
    
    def create_portfolio_tab(self):
        """Create the portfolio overview tab."""
        portfolio_frame = ttk.Frame(self.notebook)
        self.notebook.add(portfolio_frame, text="Portfolio Overview")
        
        # Portfolio Summary Section
        summary_frame = ttk.LabelFrame(portfolio_frame, text="Portfolio Summary", padding=10)
        summary_frame.pack(fill='x', padx=10, pady=5)
        
        # Cash Balance
        ttk.Label(summary_frame, text="Cash Balance:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky='w', padx=5)
        self.cash_label = ttk.Label(summary_frame, text="$0.00", font=('Arial', 12), foreground='green')
        self.cash_label.grid(row=0, column=1, sticky='w', padx=5)
        
        # Total Portfolio Value
        ttk.Label(summary_frame, text="Total Portfolio Value:", font=('Arial', 12, 'bold')).grid(row=1, column=0, sticky='w', padx=5)
        self.total_value_label = ttk.Label(summary_frame, text="$0.00", font=('Arial', 12), foreground='blue')
        self.total_value_label.grid(row=1, column=1, sticky='w', padx=5)
        
        # Total Return
        ttk.Label(summary_frame, text="Total Return:", font=('Arial', 12, 'bold')).grid(row=2, column=0, sticky='w', padx=5)
        self.return_label = ttk.Label(summary_frame, text="$0.00 (0.00%)", font=('Arial', 12))
        self.return_label.grid(row=2, column=1, sticky='w', padx=5)
        
        # Last Transaction Date
        ttk.Label(summary_frame, text="Last Transaction:", font=('Arial', 12, 'bold')).grid(row=3, column=0, sticky='w', padx=5)
        self.last_transaction_label = ttk.Label(summary_frame, text="None", font=('Arial', 12))
        self.last_transaction_label.grid(row=3, column=1, sticky='w', padx=5)
        
        # Action Buttons
        button_frame = ttk.Frame(portfolio_frame)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(button_frame, text="Refresh Portfolio", command=self.refresh_portfolio_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Execute Auto Trading", command=self.execute_auto_trading).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Generate Report", command=self.show_portfolio_report).pack(side='left', padx=5)
        
        # Recent Transactions
        transactions_frame = ttk.LabelFrame(portfolio_frame, text="Recent Transactions", padding=10)
        transactions_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Transactions Treeview
        columns = ('Date', 'Type', 'Symbol', 'Quantity', 'Price', 'Total', 'Fee')
        self.transactions_tree = ttk.Treeview(transactions_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.transactions_tree.heading(col, text=col)
            self.transactions_tree.column(col, width=100)
        
        # Scrollbar for transactions
        trans_scrollbar = ttk.Scrollbar(transactions_frame, orient='vertical', command=self.transactions_tree.yview)
        self.transactions_tree.configure(yscrollcommand=trans_scrollbar.set)
        
        self.transactions_tree.pack(side='left', fill='both', expand=True)
        trans_scrollbar.pack(side='right', fill='y')
    
    def create_trading_tab(self):
        """Create the trading tab for manual buy/sell operations."""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="Trading")
        
        # Stock Selection
        selection_frame = ttk.LabelFrame(trading_frame, text="Stock Selection", padding=10)
        selection_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(selection_frame, text="Stock Symbol:").grid(row=0, column=0, sticky='w', padx=5)
        self.symbol_entry = ttk.Entry(selection_frame, width=10)
        self.symbol_entry.grid(row=0, column=1, padx=5)
        
        ttk.Button(selection_frame, text="Get Quote", command=self.get_stock_quote).grid(row=0, column=2, padx=5)
        
        # Stock Info Display
        self.stock_info_text = scrolledtext.ScrolledText(selection_frame, height=4, width=60)
        self.stock_info_text.grid(row=1, column=0, columnspan=3, pady=5, sticky='ew')
        
        # Trading Actions
        actions_frame = ttk.LabelFrame(trading_frame, text="Trading Actions", padding=10)
        actions_frame.pack(fill='x', padx=10, pady=5)
        
        # Buy Section
        buy_frame = ttk.Frame(actions_frame)
        buy_frame.pack(side='left', fill='both', expand=True, padx=10)
        
        ttk.Label(buy_frame, text="BUY", font=('Arial', 14, 'bold'), foreground='green').pack()
        
        ttk.Label(buy_frame, text="Quantity:").pack()
        self.buy_quantity_entry = ttk.Entry(buy_frame, width=10)
        self.buy_quantity_entry.pack(pady=2)
        
        ttk.Button(buy_frame, text="Execute Buy", command=self.execute_buy, style='Buy.TButton').pack(pady=5)
        
        # Sell Section
        sell_frame = ttk.Frame(actions_frame)
        sell_frame.pack(side='right', fill='both', expand=True, padx=10)
        
        ttk.Label(sell_frame, text="SELL", font=('Arial', 14, 'bold'), foreground='red').pack()
        
        ttk.Label(sell_frame, text="Quantity:").pack()
        self.sell_quantity_entry = ttk.Entry(sell_frame, width=10)
        self.sell_quantity_entry.pack(pady=2)
        
        ttk.Button(sell_frame, text="Execute Sell", command=self.execute_sell, style='Sell.TButton').pack(pady=5)
        
        # Trading Log
        log_frame = ttk.LabelFrame(trading_frame, text="Trading Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.trading_log = scrolledtext.ScrolledText(log_frame, height=10)
        self.trading_log.pack(fill='both', expand=True)
        
        # Configure button styles
        style = ttk.Style()
        style.configure('Buy.TButton', foreground='green')
        style.configure('Sell.TButton', foreground='red')
    
    def create_holdings_tab(self):
        """Create the holdings tab to display current positions."""
        holdings_frame = ttk.Frame(self.notebook)
        self.notebook.add(holdings_frame, text="Holdings")
        
        # Holdings Treeview
        columns = ('Symbol', 'Quantity', 'Avg Cost', 'Current Price', 'Market Value', 'Gain/Loss', 'Gain/Loss %')
        self.holdings_tree = ttk.Treeview(holdings_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.holdings_tree.heading(col, text=col)
            self.holdings_tree.column(col, width=120)
        
        # Scrollbar for holdings
        holdings_scrollbar = ttk.Scrollbar(holdings_frame, orient='vertical', command=self.holdings_tree.yview)
        self.holdings_tree.configure(yscrollcommand=holdings_scrollbar.set)
        
        self.holdings_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        holdings_scrollbar.pack(side='right', fill='y', pady=10)
        
        # Holdings summary
        summary_frame = ttk.Frame(holdings_frame)
        summary_frame.pack(side='bottom', fill='x', padx=10, pady=5)
        
        ttk.Button(summary_frame, text="Refresh Holdings", command=self.refresh_holdings_data).pack(side='left', padx=5)
        ttk.Button(summary_frame, text="Export Holdings", command=self.export_holdings).pack(side='left', padx=5)
    
    def create_performance_tab(self):
        """Create the performance tab with charts."""
        performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(performance_frame, text="Performance")
        
        # Chart controls
        controls_frame = ttk.Frame(performance_frame)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(controls_frame, text="Update Chart", command=self.update_performance_chart).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Export Chart", command=self.export_chart).pack(side='left', padx=5)
        
        # Chart area
        self.chart_frame = ttk.Frame(performance_frame)
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Initialize chart
        self.create_performance_chart()
    
    def create_settings_tab(self):
        """Create the settings tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Auto-trading settings
        auto_frame = ttk.LabelFrame(settings_frame, text="Auto-Trading Settings", padding=10)
        auto_frame.pack(fill='x', padx=10, pady=5)
        
        self.auto_trading_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_frame, text="Enable Auto-Trading", variable=self.auto_trading_var).pack(anchor='w')
        
        ttk.Label(auto_frame, text="Check Interval (minutes):").pack(anchor='w')
        self.check_interval_var = tk.StringVar(value="60")
        ttk.Entry(auto_frame, textvariable=self.check_interval_var, width=10).pack(anchor='w', pady=2)
        
        # Database settings
        db_frame = ttk.LabelFrame(settings_frame, text="Database Settings", padding=10)
        db_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(db_frame, text=f"Database Path: {appconfig.DB_NAME}").pack(anchor='w')
        ttk.Button(db_frame, text="Backup Database", command=self.backup_database).pack(anchor='w', pady=5)
        
        # Status
        status_frame = ttk.LabelFrame(settings_frame, text="System Status", padding=10)
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8)
        self.status_text.pack(fill='both', expand=True)
        
        self.log_status("Portfolio GUI initialized successfully")
    
    def refresh_portfolio_data(self):
        """Refresh all portfolio data."""
        try:
            portfolio, holdings = stockSimulator.get_portfolio_state()
            
            if portfolio:
                # Update portfolio summary
                cash_balance = portfolio[1]
                total_value = stockSimulator.calculate_current_portfolio_value()
                total_return = total_value - 10000
                return_pct = (total_return / 10000) * 100 if total_value > 0 else 0
                
                self.cash_label.config(text=f"${cash_balance:,.2f}")
                self.total_value_label.config(text=f"${total_value:,.2f}")
                
                return_color = 'green' if total_return >= 0 else 'red'
                self.return_label.config(
                    text=f"${total_return:+,.2f} ({return_pct:+.2f}%)",
                    foreground=return_color
                )
                
                last_transaction = portfolio[3] if portfolio[3] else "None"
                self.last_transaction_label.config(text=last_transaction)
            
            # Update recent transactions
            self.refresh_transactions_data()
            
            # Update holdings
            self.refresh_holdings_data()
            
            self.log_status("Portfolio data refreshed successfully")
            
        except Exception as e:
            self.log_status(f"Error refreshing portfolio data: {e}")
            messagebox.showerror("Error", f"Failed to refresh portfolio data: {e}")
    
    def refresh_transactions_data(self):
        """Refresh the transactions display."""
        try:
            # Clear existing items
            for item in self.transactions_tree.get_children():
                self.transactions_tree.delete(item)
            
            # Get recent transactions
            transactions_df = db_manager.get_transactions_df()
            
            if not transactions_df.empty:
                # Show last 20 transactions
                recent_transactions = transactions_df.tail(20).iloc[::-1]  # Reverse to show newest first
                
                for _, transaction in recent_transactions.iterrows():
                    values = (
                        transaction['transaction_date'],
                        transaction['transaction_type'],
                        transaction['symbol'],
                        transaction['quantity'],
                        f"${transaction['price_per_share']:.2f}",
                        f"${transaction['total_amount']:.2f}",
                        f"${transaction['brokerage_fee']:.2f}"
                    )
                    self.transactions_tree.insert('', 'end', values=values)
        
        except Exception as e:
            self.log_status(f"Error refreshing transactions: {e}")
    
    def refresh_holdings_data(self):
        """Refresh the holdings display."""
        try:
            # Clear existing items
            for item in self.holdings_tree.get_children():
                self.holdings_tree.delete(item)
            
            portfolio, holdings = stockSimulator.get_portfolio_state()
            
            if holdings:
                # Get current prices for all holdings
                symbols = [holding[1] for holding in holdings]
                current_prices = stockSimulator.get_current_stock_prices(symbols)
                
                for holding in holdings:
                    symbol = holding[1]
                    quantity = holding[2]
                    avg_cost = holding[3]
                    total_cost = holding[4]
                    
                    if symbol in current_prices:
                        current_price = current_prices[symbol]
                        market_value = quantity * current_price
                        gain_loss = market_value - total_cost
                        gain_loss_pct = (gain_loss / total_cost) * 100 if total_cost > 0 else 0
                        
                        # Color coding for gains/losses
                        gain_loss_color = 'green' if gain_loss >= 0 else 'red'
                        
                        values = (
                            symbol,
                            quantity,
                            f"${avg_cost:.2f}",
                            f"${current_price:.2f}",
                            f"${market_value:,.2f}",
                            f"${gain_loss:+,.2f}",
                            f"{gain_loss_pct:+.2f}%"
                        )
                        
                        item = self.holdings_tree.insert('', 'end', values=values)
                        
                        # Apply color coding (note: this is basic, advanced styling would need custom widgets)
                        if gain_loss < 0:
                            self.holdings_tree.set(item, 'Gain/Loss', f"${gain_loss:,.2f}")
        
        except Exception as e:
            self.log_status(f"Error refreshing holdings: {e}")
    
    def get_stock_quote(self):
        """Get current stock quote and display information."""
        symbol = self.symbol_entry.get().upper().strip()
        
        if not symbol:
            messagebox.showwarning("Warning", "Please enter a stock symbol")
            return
        
        try:
            # Get stock info using yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            current_prices = stockSimulator.get_current_stock_prices([symbol])
            current_price = current_prices.get(symbol, "N/A")
            
            # Display stock information
            stock_info = f"Symbol: {symbol}\n"
            stock_info += f"Company: {info.get('longName', 'N/A')}\n"
            stock_info += f"Current Price: ${current_price:.2f}\n" if current_price != "N/A" else "Current Price: N/A\n"
            stock_info += f"Sector: {info.get('sector', 'N/A')}\n"
            stock_info += f"Industry: {info.get('industry', 'N/A')}\n"
            stock_info += f"Market Cap: {info.get('marketCap', 'N/A')}\n"
            
            self.stock_info_text.delete(1.0, tk.END)
            self.stock_info_text.insert(1.0, stock_info)
            
            self.log_status(f"Retrieved quote for {symbol}")
            
        except Exception as e:
            self.log_status(f"Error getting quote for {symbol}: {e}")
            messagebox.showerror("Error", f"Failed to get quote for {symbol}: {e}")
    
    def execute_buy(self):
        """Execute a buy transaction."""
        symbol = self.symbol_entry.get().upper().strip()
        quantity_str = self.buy_quantity_entry.get().strip()
        
        if not symbol or not quantity_str:
            messagebox.showwarning("Warning", "Please enter symbol and quantity")
            return
        
        try:
            quantity = int(quantity_str)
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            # Check if we can make transactions
            if not stockSimulator.can_make_transactions():
                messagebox.showwarning("Warning", "Less than a day has passed since last transaction")
                return
            
            # Get current price
            current_prices = stockSimulator.get_current_stock_prices([symbol])
            if symbol not in current_prices:
                messagebox.showerror("Error", f"Could not get current price for {symbol}")
                return
            
            current_price = current_prices[symbol]
            total_cost = (quantity * current_price) + 10  # Include brokerage fee
            
            # Check if we have enough cash
            portfolio, _ = stockSimulator.get_portfolio_state()
            cash_balance = portfolio[1] if portfolio else 0
            
            if total_cost > cash_balance:
                messagebox.showerror("Error", f"Insufficient funds. Need ${total_cost:.2f}, have ${cash_balance:.2f}")
                return
            
            # Confirm transaction
            if messagebox.askyesno("Confirm Buy", 
                                 f"Buy {quantity} shares of {symbol} at ${current_price:.2f} per share?\n"
                                 f"Total cost: ${total_cost:.2f} (including $10 fee)"):
                
                # Execute the transaction
                stock_id = db_manager.get_or_create_stock_id(symbol)
                stockSimulator.execute_buy_transaction(stock_id, symbol, quantity, current_price)
                
                self.log_trading(f"BUY: {quantity} shares of {symbol} at ${current_price:.2f}")
                self.refresh_portfolio_data()
                
                # Clear entry
                self.buy_quantity_entry.delete(0, tk.END)
                
                messagebox.showinfo("Success", f"Successfully bought {quantity} shares of {symbol}")
        
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid quantity: {e}")
        except Exception as e:
            self.log_status(f"Error executing buy: {e}")
            messagebox.showerror("Error", f"Failed to execute buy: {e}")
    
    def execute_sell(self):
        """Execute a sell transaction."""
        symbol = self.symbol_entry.get().upper().strip()
        quantity_str = self.sell_quantity_entry.get().strip()
        
        if not symbol or not quantity_str:
            messagebox.showwarning("Warning", "Please enter symbol and quantity")
            return
        
        try:
            quantity = int(quantity_str)
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            # Check if we can make transactions
            if not stockSimulator.can_make_transactions():
                messagebox.showwarning("Warning", "Less than a day has passed since last transaction")
                return
            
            # Check if we own this stock
            portfolio, holdings = stockSimulator.get_portfolio_state()
            
            owned_quantity = 0
            stock_id = None
            for holding in holdings:
                if holding[1] == symbol:  # symbol is at index 1
                    owned_quantity = holding[2]  # quantity is at index 2
                    stock_id = holding[0]  # stock_id is at index 0
                    break
            
            if owned_quantity == 0:
                messagebox.showerror("Error", f"You don't own any shares of {symbol}")
                return
            
            if quantity > owned_quantity:
                messagebox.showerror("Error", f"You only own {owned_quantity} shares of {symbol}")
                return
            
            # Get current price
            current_prices = stockSimulator.get_current_stock_prices([symbol])
            if symbol not in current_prices:
                messagebox.showerror("Error", f"Could not get current price for {symbol}")
                return
            
            current_price = current_prices[symbol]
            total_proceeds = (quantity * current_price) - 10  # Subtract brokerage fee
            
            # Confirm transaction
            if messagebox.askyesno("Confirm Sell", 
                                 f"Sell {quantity} shares of {symbol} at ${current_price:.2f} per share?\n"
                                 f"Total proceeds: ${total_proceeds:.2f} (after $10 fee)"):
                
                # Execute the transaction
                stockSimulator.execute_sell_transaction(stock_id, symbol, quantity, current_price)
                
                self.log_trading(f"SELL: {quantity} shares of {symbol} at ${current_price:.2f}")
                self.refresh_portfolio_data()
                
                # Clear entry
                self.sell_quantity_entry.delete(0, tk.END)
                
                messagebox.showinfo("Success", f"Successfully sold {quantity} shares of {symbol}")
        
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid quantity: {e}")
        except Exception as e:
            self.log_status(f"Error executing sell: {e}")
            messagebox.showerror("Error", f"Failed to execute sell: {e}")
    
    def execute_auto_trading(self):
        """Execute automatic trading based on analysis."""
        try:
            if not stockSimulator.can_make_transactions():
                messagebox.showinfo("Info", "Less than a day has passed since last transaction. No auto-trading executed.")
                return
            
            # Run in separate thread to avoid blocking GUI
            def run_auto_trading():
                try:
                    self.log_status("Starting automatic trading analysis...")
                    stockSimulator.execute_portfolio_transactions()
                    
                    # Update GUI in main thread
                    self.root.after(0, lambda: [
                        self.refresh_portfolio_data(),
                        self.log_status("Automatic trading completed successfully")
                    ])
                    
                except Exception as e:
                    self.root.after(0, lambda: [
                        self.log_status(f"Error in automatic trading: {e}"),
                        messagebox.showerror("Error", f"Automatic trading failed: {e}")
                    ])
            
            threading.Thread(target=run_auto_trading, daemon=True).start()
            
        except Exception as e:
            self.log_status(f"Error starting auto trading: {e}")
            messagebox.showerror("Error", f"Failed to start auto trading: {e}")
    
    def show_portfolio_report(self):
        """Show detailed portfolio report."""
        try:
            report = stockSimulator.get_portfolio_report()
            
            # Create new window for report
            report_window = tk.Toplevel(self.root)
            report_window.title("Portfolio Report")
            report_window.geometry("600x400")
            
            report_text = scrolledtext.ScrolledText(report_window, font=('Courier', 10))
            report_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            report_text.insert(1.0, report)
            report_text.config(state='disabled')
            
        except Exception as e:
            self.log_status(f"Error generating report: {e}")
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    def create_performance_chart(self):
        """Create the performance chart."""
        try:
            # Create matplotlib figure
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
            self.fig.suptitle('Portfolio Performance', fontsize=16)
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Initial chart update
            self.update_performance_chart()
            
        except Exception as e:
            self.log_status(f"Error creating chart: {e}")
    
    def update_performance_chart(self):
        """Update the performance chart with current data."""
        try:
            # Clear axes
            self.ax1.clear()
            self.ax2.clear()
            
            # Get portfolio states
            conn = sqlite3.connect(appconfig.DB_NAME)
            portfolio_states_df = pd.read_sql_query('''
                SELECT * FROM portfolio_state 
                ORDER BY created_date ASC
            ''', conn)
            conn.close()
            
            if portfolio_states_df.empty:
                self.ax1.text(0.5, 0.5, 'No portfolio data available', 
                             ha='center', va='center', transform=self.ax1.transAxes)
                self.canvas.draw()
                return
            
            # Convert dates
            portfolio_states_df['created_date'] = pd.to_datetime(portfolio_states_df['created_date'])
            
            # Get transactions for holdings calculation
            transactions_df = db_manager.get_transactions_df()
            
            # Calculate portfolio values over time
            dates = []
            portfolio_values = []
            cash_values = []
            holdings_values = []
            
            for _, state in portfolio_states_df.iterrows():
                date = state['created_date']
                cash = state['cash_balance']
                
                # Calculate holdings value for this date
                holdings, holdings_value = stockSimulator.reconstruct_holdings_and_value(date, transactions_df)
                
                dates.append(date)
                cash_values.append(cash)
                holdings_values.append(holdings_value)
                portfolio_values.append(cash + holdings_value)
            
            # Plot portfolio value
            self.ax1.plot(dates, portfolio_values, 'b-', linewidth=2, label='Total Portfolio Value')
            self.ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.7, label='Initial Investment')
            self.ax1.set_ylabel('Portfolio Value ($)')
            self.ax1.set_title('Portfolio Value Over Time')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            
            # Plot composition
            self.ax2.plot(dates, cash_values, 'g-', label='Cash', linewidth=2)
            self.ax2.plot(dates, holdings_values, 'orange', label='Holdings', linewidth=2)
            self.ax2.set_ylabel('Value ($)')
            self.ax2.set_xlabel('Date')
            self.ax2.set_title('Portfolio Composition')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            
            # Format dates on x-axis
            self.fig.autofmt_xdate()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            self.log_status(f"Error updating chart: {e}")
    
    def export_chart(self):
        """Export the performance chart."""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.log_status(f"Chart exported to {filename}")
                messagebox.showinfo("Success", f"Chart exported to {filename}")
        except Exception as e:
            self.log_status(f"Error exporting chart: {e}")
            messagebox.showerror("Error", f"Failed to export chart: {e}")
    
    def export_holdings(self):
        """Export holdings data to CSV."""
        try:
            from tkinter import filedialog
            import csv
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                portfolio, holdings = stockSimulator.get_portfolio_state()
                
                if holdings:
                    symbols = [holding[1] for holding in holdings]
                    current_prices = stockSimulator.get_current_stock_prices(symbols)
                    
                    with open(filename, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['Symbol', 'Quantity', 'Avg Cost', 'Current Price', 'Market Value', 'Gain/Loss', 'Gain/Loss %'])
                        
                        for holding in holdings:
                            symbol = holding[1]
                            quantity = holding[2]
                            avg_cost = holding[3]
                            total_cost = holding[4]
                            
                            if symbol in current_prices:
                                current_price = current_prices[symbol]
                                market_value = quantity * current_price
                                gain_loss = market_value - total_cost
                                gain_loss_pct = (gain_loss / total_cost) * 100 if total_cost > 0 else 0
                                
                                writer.writerow([
                                    symbol, quantity, f"{avg_cost:.2f}", f"{current_price:.2f}",
                                    f"{market_value:.2f}", f"{gain_loss:.2f}", f"{gain_loss_pct:.2f}"
                                ])
                    
                    self.log_status(f"Holdings exported to {filename}")
                    messagebox.showinfo("Success", f"Holdings exported to {filename}")
                else:
                    messagebox.showinfo("Info", "No holdings to export")
        
        except Exception as e:
            self.log_status(f"Error exporting holdings: {e}")
            messagebox.showerror("Error", f"Failed to export holdings: {e}")
    
    def backup_database(self):
        """Create a backup of the database."""
        try:
            from tkinter import filedialog
            import shutil
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".db",
                filetypes=[("Database files", "*.db"), ("All files", "*.*")]
            )
            if filename:
                shutil.copy2(appconfig.DB_NAME, filename)
                self.log_status(f"Database backed up to {filename}")
                messagebox.showinfo("Success", f"Database backed up to {filename}")
        
        except Exception as e:
            self.log_status(f"Error backing up database: {e}")
            messagebox.showerror("Error", f"Failed to backup database: {e}")
    
    def start_background_updates(self):
        """Start background thread for automatic updates."""
        def background_worker():
            while True:
                try:
                    if self.auto_trading_var.get():
                        # Check if it's been a day since last update
                        now = datetime.now(TIME_ZONE)
                        time_diff = now - self.last_update_check
                        
                        if time_diff.total_seconds() >= 24 * 60 * 60:  # 24 hours
                            # Check if we can make transactions
                            if stockSimulator.can_make_transactions():
                                self.root.after(0, lambda: [
                                    self.log_status("Auto-update: Executing portfolio transactions..."),
                                    self.execute_auto_trading()
                                ])
                            
                            self.last_update_check = now
                    
                    # Sleep for the specified interval
                    interval_minutes = int(self.check_interval_var.get())
                    time.sleep(interval_minutes * 60)
                
                except Exception as e:
                    self.root.after(0, lambda: self.log_status(f"Background update error: {e}"))
                    time.sleep(300)  # Sleep 5 minutes on error
        
        # Start background thread
        bg_thread = threading.Thread(target=background_worker, daemon=True)
        bg_thread.start()
    
    def log_status(self, message):
        """Log a status message."""
        timestamp = datetime.now(TIME_ZONE).strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, log_message)
        self.status_text.see(tk.END)
        
        # Keep only last 100 lines
        lines = self.status_text.get(1.0, tk.END).split('\n')
        if len(lines) > 100:
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(1.0, '\n'.join(lines[-100:]))
    
    def log_trading(self, message):
        """Log a trading message."""
        timestamp = datetime.now(TIME_ZONE).strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.trading_log.insert(tk.END, log_message)
        self.trading_log.see(tk.END)
        
        # Keep only last 50 lines
        lines = self.trading_log.get(1.0, tk.END).split('\n')
        if len(lines) > 50:
            self.trading_log.delete(1.0, tk.END)
            self.trading_log.insert(1.0, '\n'.join(lines[-50:]))


def main():
    """Main function to run the Portfolio GUI."""
    root = tk.Tk()
    app = PortfolioGUI(root)
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit the Portfolio Manager?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()



"""
Web UI Demo - Complete Stock Analyzer with New Architecture

This creates a Flask web application that:
1. Runs the stock analysis using the new architecture
2. Generates all plots and data
3. Provides a web interface to view results
4. Shows database statistics and file locations
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any
from flask import Flask, render_template_string, send_from_directory, jsonify, request
import threading
import webbrowser
import time

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import the new architecture
from shared.config import get_settings, setup_logging
from shared.logging import get_logger
from domain.entities.stock import Stock, StockPrice, StockInfo
from infrastructure.database import SqliteStockRepository

# Import portfolio simulator
try:
    from legacy.notcurrentlyused.simple_portfolio_simulator import SimplePortfolioSimulator
    PORTFOLIO_ENABLED = True
    print("Portfolio simulator loaded successfully!")
    
    # Initialize global simulator instance
    portfolio_simulator = SimplePortfolioSimulator(db_path="data/web_demo_portfolio.db")
except ImportError as e:
    print(f"Portfolio simulator not available: {e}")
    PORTFOLIO_ENABLED = False
    portfolio_simulator = None
except Exception as e:
    print(f"Error loading portfolio simulator: {e}")
    PORTFOLIO_ENABLED = False
    portfolio_simulator = None

print(f"PORTFOLIO_ENABLED: {PORTFOLIO_ENABLED}")

# Sector color mapping
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

class WebStockAnalyzer:
    """Web-enabled stock analyzer using the new architecture."""
    
    def __init__(self, db_path: str = "data/web_demo.db"):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.repo = SqliteStockRepository(db_path=db_path)
        
        # Create plots directory
        self.plots_path = "plots/web_demo"
        os.makedirs(self.plots_path, exist_ok=True)
        
        # Analysis results
        self.analysis_complete = False
        self.buy_candidates = []
        self.sell_candidates = []
        self.stocks_data = []
        self.database_stats = {}
    
    def get_nasdaq_100_tickers(self) -> List[str]:
        """Scrape NASDAQ-100 tickers from Wikipedia."""
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')
            tickers = tables[4]['Ticker'].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
            return tickers
        except Exception as e:
            self.logger.error(f"Error scraping NASDAQ-100 tickers: {e}")
            return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    
    async def sync_ticker_data(self, symbol: str) -> bool:
        """Sync data for a single ticker."""
        try:
            stock = await self.repo.get_stock_by_symbol(symbol)
            if not stock:
                stock = await self.repo.create_stock(Stock(
                    id=None, symbol=symbol, name=None, exchange=None
                ))
            
            latest_date = await self.repo.get_latest_stock_price_date(stock.id)
            start_date = '2023-01-01'
            if latest_date:
                start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(start=start_date)
            
            if hist.empty:
                return True
            
            prices = []
            for date, row in hist.iterrows():
                if pd.isna(row[['Open', 'High', 'Low', 'Close', 'Volume']]).any():
                    continue
                
                price = StockPrice(
                    id=None, stock_id=stock.id, date=date.date(),
                    open_price=Decimal(str(row['Open'])),
                    high_price=Decimal(str(row['High'])),
                    low_price=Decimal(str(row['Low'])),
                    close_price=Decimal(str(row['Close'])),
                    volume=int(row['Volume']),
                    adjusted_close=Decimal(str(row['Close']))
                )
                prices.append(price)
            
            if prices:
                await self.repo.create_stock_prices(prices)
            
            # Get stock info
            try:
                info = ticker_obj.info
                if info and 'sector' in info:
                    stock_info = StockInfo(
                        id=None, stock_id=stock.id,
                        sector=info.get('sector'),
                        industry=info.get('industry'),
                        market_cap=Decimal(str(info.get('marketCap', 0))) if info.get('marketCap') else None,
                        pe_ratio=Decimal(str(info.get('trailingPE', 0))) if info.get('trailingPE') else None,
                        dividend_yield=Decimal(str(info.get('dividendYield', 0))) if info.get('dividendYield') else None,
                        beta=Decimal(str(info.get('beta', 0))) if info.get('beta') else None,
                        description=info.get('longBusinessSummary'),
                        website=info.get('website'),
                        employees=info.get('fullTimeEmployees')
                    )
                    await self.repo.create_or_update_stock_info(stock_info)
            except:
                pass
            
            return True
        except Exception as e:
            self.logger.error(f"Error syncing {symbol}: {e}")
            return False
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_volatility(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate rolling volatility."""
        returns = prices.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)
        return volatility.fillna(0)
    
    async def get_stock_dataframe(self) -> pd.DataFrame:
        """Get stock data as DataFrame for analysis."""
        all_stocks = await self.repo.get_all_stocks()
        all_data = []
        
        for stock in all_stocks:
            prices = await self.repo.get_stock_prices(stock.id, limit=200)
            if len(prices) < 50:
                continue
            
            stock_info = await self.repo.get_stock_info(stock.id)
            
            for price in prices:
                all_data.append({
                    'symbol': stock.symbol,
                    'stock_id': stock.id,
                    'date': price.date,
                    'open': float(price.open_price),
                    'high': float(price.high_price),
                    'low': float(price.low_price),
                    'close': float(price.close_price),
                    'volume': price.volume,
                    'sector': stock_info.sector if stock_info else 'Unknown',
                    'industry': stock_info.industry if stock_info else 'Unknown'
                })
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['symbol', 'date'])
        
        return df
    
    def analyze_ticker(self, df_ticker: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a single ticker for buy/sell signals."""
        if len(df_ticker) < 50:
            return None
        
        close = df_ticker['close']
        volume = df_ticker['volume']
        
        ma20 = close.rolling(window=20).mean()
        ma50 = close.rolling(window=50).mean()
        rsi = self.calculate_rsi(close)
        volatility = self.calculate_volatility(close)
        momentum = close - close.shift(7)
        vol_change = volume.rolling(window=5).mean() - volume.rolling(window=5).mean().shift(5)
        
        latest_idx = -1
        metrics = {
            'close': close.iloc[latest_idx],
            'ma20': ma20.iloc[latest_idx],
            'ma50': ma50.iloc[latest_idx],
            'rsi': rsi.iloc[latest_idx],
            'momentum': momentum.iloc[latest_idx],
            'vol_change': vol_change.iloc[latest_idx],
            'volatility': volatility.iloc[latest_idx],
        }
        
        buy_score = sum([
            metrics['close'] > metrics['ma20'],
            metrics['close'] > metrics['ma50'],
            metrics['rsi'] < 40,
            metrics['momentum'] > 0,
            metrics['vol_change'] > 0,
            metrics['volatility'] <= 0.3
        ])
        
        sell_score = sum([
            metrics['close'] < metrics['ma20'],
            metrics['close'] < metrics['ma50'],
            metrics['rsi'] > 60,
            metrics['momentum'] < 0,
            metrics['vol_change'] < 0,
            metrics['volatility'] > 0.5
        ])
        
        return {
            'buy_score': buy_score,
            'sell_score': sell_score,
            'volatility': metrics['volatility'],
            'sector': df_ticker['sector'].iloc[0],
            'industry': df_ticker['industry'].iloc[0],
            'current_price': metrics['close'],
            'rsi': metrics['rsi']
        }
    
    def create_stock_plot(self, df_ticker: pd.DataFrame, ticker: str):
        """Create detailed stock analysis plot."""
        df_ticker = df_ticker.copy()
        df_ticker['MA20'] = df_ticker['close'].rolling(window=20).mean()
        df_ticker['MA50'] = df_ticker['close'].rolling(window=50).mean()
        df_ticker['RSI'] = self.calculate_rsi(df_ticker['close'])
        df_ticker['Volatility'] = self.calculate_volatility(df_ticker['close'])
        
        sector = df_ticker['sector'].iloc[0]
        industry = df_ticker['industry'].iloc[0]
        sector_color = SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])
        
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            subplot_titles=(f'{ticker} - {sector} ({industry})', 'RSI', 'Volatility', 'Volume'),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Price and MAs
        fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker['close'],
                                mode='lines', name='Close Price',
                                line=dict(color=sector_color, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker['MA20'],
                                mode='lines', name='20-day MA',
                                line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker['MA50'],
                                mode='lines', name='50-day MA',
                                line=dict(color='green', width=1)), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker['RSI'],
                                mode='lines', name='RSI',
                                line=dict(color='purple', width=1)), row=2, col=1)
        
        # Volatility
        fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker['Volatility'],
                                mode='lines', name='Volatility',
                                line=dict(color='red', width=1)), row=3, col=1)
        
        # Volume
        fig.add_trace(go.Bar(x=df_ticker['date'], y=df_ticker['volume'],
                            name='Volume', marker_color='lightgray'), row=4, col=1)
        
        fig.update_layout(
            title=f'{ticker} Stock Analysis - {sector} Sector',
            height=900, showlegend=True, hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Volatility", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)
        
        filename = os.path.join(self.plots_path, f"{ticker}_analysis.html")
        fig.write_html(filename)
        return filename
    
    def create_sector_overview_plot(self):
        """Create sector overview plot."""
        all_candidates = []
        
        for ticker, score, volatility, sector, industry, price, rsi in self.buy_candidates:
            all_candidates.append({
                'ticker': ticker, 'score': score, 'volatility': volatility,
                'sector': sector, 'industry': industry, 'recommendation': 'Buy',
                'price': price, 'rsi': rsi,
                'risk_level': 'Low' if volatility < 0.3 else 'Medium' if volatility < 0.5 else 'High'
            })
        
        for ticker, score, volatility, sector, industry, price, rsi in self.sell_candidates:
            all_candidates.append({
                'ticker': ticker, 'score': score, 'volatility': volatility,
                'sector': sector, 'industry': industry, 'recommendation': 'Sell',
                'price': price, 'rsi': rsi,
                'risk_level': 'Low' if volatility < 0.3 else 'Medium' if volatility < 0.5 else 'High'
            })
        
        if not all_candidates:
            return None
        
        df_candidates = pd.DataFrame(all_candidates)
        
        fig = px.scatter(
            df_candidates, x='rsi', y='score', color='sector', symbol='recommendation',
            size='volatility', hover_data=['ticker', 'industry', 'price', 'risk_level'],
            title='Stock Recommendations: RSI vs Score (Size = Volatility)',
            color_discrete_map=SECTOR_COLORS, size_max=20
        )
        
        fig.update_layout(height=700, showlegend=True)
        filename = os.path.join(self.plots_path, "sector_overview.html")
        fig.write_html(filename)
        return filename
    
    async def run_analysis(self):
        """Run the complete stock analysis."""
        self.logger.info("Starting web demo analysis")
        
        # Get tickers
        nasdaq_tickers = self.get_nasdaq_100_tickers()
        demo_tickers = nasdaq_tickers[:15]  # Limit for demo
        
        # Sync data
        for ticker in demo_tickers:
            await self.sync_ticker_data(ticker)
        
        # Get data for analysis
        df = await self.get_stock_dataframe()
        if df.empty:
            return
        
        # Analyze stocks
        self.buy_candidates = []
        self.sell_candidates = []
        
        grouped = df.groupby('symbol')
        for ticker, group in grouped:
            result = self.analyze_ticker(group)
            if result is None:
                continue
            
            candidate_data = (
                ticker, result['buy_score'], result['volatility'],
                result['sector'], result['industry'],
                result['current_price'], result['rsi']
            )
            
            if result['buy_score'] >= 3:
                self.buy_candidates.append(candidate_data)
            
            if result['sell_score'] >= 3:
                self.sell_candidates.append(candidate_data)
        
        # Sort candidates
        self.buy_candidates.sort(key=lambda x: x[1], reverse=True)
        self.sell_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Create plots
        self.create_sector_overview_plot()
        
        # Create individual plots for top stocks
        top_stocks = []
        if self.buy_candidates:
            top_stocks.extend([t[0] for t in self.buy_candidates[:3]])
        if self.sell_candidates:
            top_stocks.extend([t[0] for t in self.sell_candidates[:2]])
        
        for ticker in top_stocks:
            if ticker in grouped.groups:
                ticker_data = grouped.get_group(ticker)
                self.create_stock_plot(ticker_data, ticker)
        
        # Get database stats
        all_stocks = await self.repo.get_all_stocks()
        sectors = await self.repo.get_sectors()
        
        self.database_stats = {
            'total_stocks': len(all_stocks),
            'total_sectors': len(sectors),
            'sectors': sectors,
            'data_points': len(df),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        }
        
        self.analysis_complete = True
        self.logger.info("Web demo analysis completed")

# Global analyzer instance
analyzer = WebStockAnalyzer()

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Main dashboard page."""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Analyzer - New Architecture Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
            .status { padding: 15px; border-radius: 5px; margin: 20px 0; }
            .status.loading { background-color: #fff3cd; border: 1px solid #ffeaa7; }
            .status.complete { background-color: #d4edda; border: 1px solid #c3e6cb; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
            .card { background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; }
            .card h3 { margin-top: 0; color: #495057; }
            .btn { display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
            .btn:hover { background-color: #0056b3; }
            .btn.success { background-color: #28a745; }
            .btn.warning { background-color: #ffc107; color: #212529; }
            .recommendations { margin: 20px 0; }
            .buy { color: #28a745; font-weight: bold; }
            .sell { color: #dc3545; font-weight: bold; }
            .file-list { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .file-list ul { margin: 0; padding-left: 20px; }
            .file-list li { margin: 5px 0; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .stat-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
            .stat-number { font-size: 2em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; }
        </style>
        <script>
            function checkStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        if (data.complete) {
                            location.reload();
                        }
                    });
            }
            
            function startAnalysis() {
                document.getElementById('start-btn').style.display = 'none';
                document.getElementById('loading-status').style.display = 'block';
                
                fetch('/api/start-analysis', {method: 'POST'})
                    .then(() => {
                        // Check status every 5 seconds
                        setInterval(checkStatus, 5000);
                    });
            }
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Stock Analyzer - New Architecture Demo</h1>
                <p>Complete refactored system with clean architecture, database integration, and interactive visualizations</p>
            </div>
            
            {% if not analyzer.analysis_complete %}
            <div class="status loading" id="loading-status" style="display: none;">
                <h3>🔄 Analysis in Progress...</h3>
                <p>Scraping tickers, fetching data, running analysis, and generating plots. This may take a few minutes.</p>
                <p><em>The page will automatically refresh when complete.</em></p>
            </div>
            
            <div style="text-align: center;">
                <button id="start-btn" class="btn" onclick="startAnalysis()">
                    🎯 Start Stock Analysis Demo
                </button>
                <p><em>This will scrape NASDAQ-100 tickers, fetch real data, and generate interactive plots</em></p>
            </div>
            {% else %}
            
            <div class="status complete">
                <h3>✅ Analysis Complete!</h3>
                <p>Successfully analyzed {{ analyzer.database_stats.total_stocks }} stocks with {{ analyzer.database_stats.data_points }} data points</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number">{{ analyzer.database_stats.total_stocks }}</div>
                    <div class="stat-label">Stocks Analyzed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{{ analyzer.buy_candidates|length }}</div>
                    <div class="stat-label">Buy Signals</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{{ analyzer.sell_candidates|length }}</div>
                    <div class="stat-label">Sell Signals</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{{ analyzer.database_stats.total_sectors }}</div>
                    <div class="stat-label">Sectors</div>
                </div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>📈 Buy Recommendations</h3>
                    {% if analyzer.buy_candidates %}
                    {% for ticker, score, vol, sector, industry, price, rsi in analyzer.buy_candidates[:5] %}
                    <div class="buy">
                        {{ ticker }} (Score: {{ score }}) - ${{ "%.2f"|format(price) }}<br>
                        <small>{{ sector }} | Vol: {{ "%.3f"|format(vol) }} | RSI: {{ "%.1f"|format(rsi) }}</small>
                    </div>
                    {% endfor %}
                    {% else %}
                    <p><em>No buy signals found</em></p>
                    {% endif %}
                </div>
                
                <div class="card">
                    <h3>📉 Sell Recommendations</h3>
                    {% if analyzer.sell_candidates %}
                    {% for ticker, score, vol, sector, industry, price, rsi in analyzer.sell_candidates[:5] %}
                    <div class="sell">
                        {{ ticker }} (Score: {{ score }}) - ${{ "%.2f"|format(price) }}<br>
                        <small>{{ sector }} | Vol: {{ "%.3f"|format(vol) }} | RSI: {{ "%.1f"|format(rsi) }}</small>
                    </div>
                    {% endfor %}
                    {% else %}
                    <p><em>No sell signals found</em></p>
                    {% endif %}
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Interactive Visualizations & Portfolio</h3>
                <p>Click the links below to view the generated interactive plots:</p>
                <a href="/plots/web_demo/sector_overview.html" target="_blank" class="btn success">
                    🎯 Sector Overview Plot
                </a>
                <a href="/browse" target="_blank" class="btn">
                    📁 Browse All Plots
                </a>
                {% if PORTFOLIO_ENABLED %}
                <a href="/portfolio" class="btn warning">
                    💼 Portfolio Management
                </a>
                {% endif %}
            </div>
            
            <div class="card">
                <h3>🗄️ Database Information</h3>
                <p><strong>Database Path:</strong> <code>{{ analyzer.repo.db_path }}</code></p>
                <p><strong>Date Range:</strong> {{ analyzer.database_stats.date_range }}</p>
                <p><strong>Sectors:</strong> {{ analyzer.database_stats.sectors|join(', ') }}</p>
            </div>
            
            <div class="card">
                <h3>📁 Generated Files</h3>
                <div class="file-list">
                    <h4>Database:</h4>
                    <ul>
                        <li>📄 <code>{{ analyzer.repo.db_path }}</code> - SQLite database with stock data</li>
                    </ul>
                    
                    <h4>Interactive Plots:</h4>
                    <ul>
                        <li>📊 <a href="/plots/web_demo/sector_overview.html" target="_blank">sector_overview.html</a> - Sector analysis</li>
                        {% for ticker, _, _, _, _, _, _ in analyzer.buy_candidates[:3] %}
                        <li>📈 <a href="/plots/web_demo/{{ ticker }}_analysis.html" target="_blank">{{ ticker }}_analysis.html</a> - Individual stock analysis</li>
                        {% endfor %}
                    </ul>
                    
                    <h4>Architecture Files:</h4>
                    <ul>
                        <li>📁 <code>domain/</code> - Business entities and services</li>
                        <li>📁 <code>infrastructure/</code> - Database repositories</li>
                        <li>📁 <code>shared/</code> - Configuration, logging, exceptions</li>
                    </ul>
                </div>
            </div>
            
            <div class="card">
                <h3>🏗️ Architecture Benefits</h3>
                <ul>
                    <li>✅ <strong>Type Safety:</strong> 100% type annotated with validation</li>
                    <li>✅ <strong>Clean Architecture:</strong> Proper separation of concerns</li>
                    <li>✅ <strong>Database Layer:</strong> Repository pattern with SQLite</li>
                    <li>✅ <strong>Configuration:</strong> Environment-based settings</li>
                    <li>✅ <strong>Error Handling:</strong> Structured exception hierarchy</li>
                    <li>✅ <strong>Logging:</strong> Professional logging with context</li>
                    <li>✅ <strong>Testing:</strong> Interface-based design for mocking</li>
                </ul>
            </div>
            
            {% endif %}
        </div>
    </body>
    </html>
    '''
    
    return render_template_string(html_template, analyzer=analyzer)

@app.route('/api/status')
def api_status():
    """API endpoint to check analysis status."""
    return jsonify({
        'complete': analyzer.analysis_complete,
        'buy_count': len(analyzer.buy_candidates),
        'sell_count': len(analyzer.sell_candidates)
    })

@app.route('/api/start-analysis', methods=['POST'])
def api_start_analysis():
    """API endpoint to start analysis."""
    def run_analysis():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(analyzer.run_analysis())
        loop.close()
    
    # Run analysis in background thread
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/browse/')
@app.route('/browse/<path:subpath>')
def browse_plots(subpath=''):
    """Browse plots directory with enhanced functionality."""
    plots_dir = os.path.abspath('plots')
    full_path = os.path.join(plots_dir, subpath) if subpath else plots_dir
    
    if not os.path.exists(full_path):
        return "Directory not found", 404
    
    if os.path.isfile(full_path):
        # If it's a file, serve it
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        return send_from_directory(directory, filename)
    
    # List directory contents
    try:
        items = os.listdir(full_path)
    except PermissionError:
        return "Permission denied", 403
    
    # Separate directories and files
    directories = []
    files = []
    
    for item in items:
        item_path = os.path.join(full_path, item)
        if os.path.isdir(item_path):
            directories.append(item)
        else:
            files.append(item)
    
    # Sort them
    directories.sort()
    files.sort()
    
    # Build navigation breadcrumbs
    breadcrumbs = []
    if subpath:
        parts = subpath.split('/')
        current_path = ''
        for part in parts:
            current_path = os.path.join(current_path, part).replace('\\', '/')
            breadcrumbs.append(f'<a href="/browse/{current_path}">{part}</a>')
    
    breadcrumb_html = ' / '.join(['<a href="/browse/">📁 Plots</a>'] + breadcrumbs)
    
    # Build directory links
    dir_links = []
    for dirname in directories:
        dir_path = os.path.join(subpath, dirname).replace('\\', '/') if subpath else dirname
        dir_links.append(f'<li class="dir-item">📁 <a href="/browse/{dir_path}"><strong>{dirname}/</strong></a></li>')
    
    # Build file links
    file_links = []
    for filename in files:
        file_path = os.path.join(subpath, filename).replace('\\', '/') if subpath else filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Choose appropriate icon
        if file_ext in ['.html']:
            icon = '📊'
        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
            icon = '🖼️'
        else:
            icon = '📄'
        
        file_links.append(f'<li class="file-item">{icon} <a href="/plots/{file_path}" target="_blank">{filename}</a></li>')
    
    # Combine all links
    all_links = dir_links + file_links
    
    # Add back navigation if in subdirectory
    back_link = ''
    if subpath:
        parent_parts = subpath.split('/')[:-1]
        parent_path = '/'.join(parent_parts) if parent_parts else ''
        back_link = f'<p><a href="/browse/{parent_path}" class="back-link">⬅️ Back to parent directory</a></p>'
    
    current_dir = f" - {subpath}" if subpath else ""
    
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Browse Plots{current_dir}</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5; 
            }}
            .container {{ 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }}
            .header {{
                text-align: center;
                color: #333;
                border-bottom: 2px solid #007bff;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .breadcrumbs {{
                background: #e9ecef;
                padding: 10px 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            .back-link {{
                display: inline-block;
                padding: 8px 15px;
                background: #6c757d;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .back-link:hover {{ background: #5a6268; }}
            ul {{ 
                list-style-type: none; 
                padding-left: 0; 
                margin: 0;
            }}
            li {{ 
                margin: 8px 0; 
                padding: 12px 15px; 
                border-radius: 5px; 
                transition: background-color 0.2s;
            }}
            .dir-item {{ 
                background: #e3f2fd; 
                border-left: 4px solid #2196f3;
            }}
            .file-item {{ 
                background: #f8f9fa; 
                border-left: 4px solid #28a745;
            }}
            li:hover {{ 
                background: #dee2e6; 
            }}
            a {{ 
                text-decoration: none; 
                color: #007bff; 
                font-weight: 500;
            }}
            a:hover {{ 
                text-decoration: underline; 
            }}
            .stats {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                text-align: center;
                color: #6c757d;
            }}
            .empty {{
                text-align: center;
                color: #6c757d;
                font-style: italic;
                padding: 40px;
            }}
            .nav-links {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .nav-links a {{
                display: inline-block;
                padding: 8px 15px;
                background: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 0 5px;
            }}
            .nav-links a:hover {{ background: #0056b3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Stock Analysis Plots Browser</h1>
                <p>Browse all generated plots and visualizations</p>
            </div>
            
            <div class="nav-links">
                <a href="/">🏠 Main Dashboard</a>
                <a href="/browse/">📁 All Plots</a>
            </div>
            
            <div class="breadcrumbs">
                <strong>📍 Location:</strong> {breadcrumbs}
            </div>
            
            {back_link}
            
            {content}
            
            <div class="stats">
                📊 {dir_count} directories • 📄 {file_count} files
            </div>
        </div>
    </body>
    </html>
    '''
    
    if all_links:
        content = f'<ul>{"".join(all_links)}</ul>'
    else:
        content = '<div class="empty">📭 This directory is empty</div>'
    
    return html_template.format(
        current_dir=current_dir,
        breadcrumbs=breadcrumb_html,
        back_link=back_link,
        content=content,
        dir_count=len(directories),
        file_count=len(files)
    )

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve plot files."""
    plots_dir = os.path.abspath('plots')
    return send_from_directory(plots_dir, filename)

# Portfolio Management Routes
if PORTFOLIO_ENABLED:
    
    @app.route('/portfolio')
    def portfolio_dashboard():
        """Portfolio dashboard page."""
        summary = portfolio_simulator.get_portfolio_summary()
        holdings = portfolio_simulator._get_holdings()
        
        # Get recent transactions
        import sqlite3
        transactions = []
        try:
            with sqlite3.connect(portfolio_simulator.db_path) as conn:
                cursor = conn.execute("SELECT * FROM transactions ORDER BY transaction_date DESC LIMIT 10")
                for row in cursor.fetchall():
                    transactions.append({
                        'date': row[7],
                        'type': row[2],
                        'symbol': row[1],
                        'quantity': row[3],
                        'price': row[4],
                        'total': row[5],
                        'fee': row[6]
                    })
        except Exception as e:
            print(f"Error getting transactions: {e}")
        
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
                .summary { background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .positive { color: green; }
                .negative { color: red; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .button { background: #007bff; color: white; padding: 10px 15px; text-decoration: none; border-radius: 3px; margin-right: 10px; }
                .button:hover { background: #0056b3; }
                .error { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📈 Portfolio Dashboard</h1>
                    <p><a href="/">← Back to Home</a> | <a href="/portfolio/trading">Trading</a> | <a href="/portfolio/performance">Performance</a></p>
                </div>
                
                <div class="summary">
                    <h2>Portfolio Summary</h2>
                    {% if summary.error %}
                        <p class="error">{{ summary.error }}</p>
                    {% else %}
                        <p><strong>Cash Balance:</strong> ${{ "%.2f"|format(summary.cash_balance) }}</p>
                        <p><strong>Total Portfolio Value:</strong> ${{ "%.2f"|format(summary.current_value) }}</p>
                        <p><strong>Total Return:</strong> 
                            <span class="{{ 'positive' if summary.total_return >= 0 else 'negative' }}">
                                ${{ "%.2f"|format(summary.total_return) }} ({{ "%.2f"|format(summary.return_percentage) }}%)
                            </span>
                        </p>
                        <p><strong>Holdings:</strong> {{ summary.holdings_count }} positions</p>
                        <p><strong>Can Trade:</strong> {{ "Yes" if summary.can_trade else "No (wait 24 hours)" }}</p>
                        <p><strong>Last Transaction:</strong> {{ summary.last_transaction_date or "None" }}</p>
                    {% endif %}
                </div>
                
                <div style="margin-bottom: 20px;">
                    <a href="/portfolio/trading" class="button">Manual Trading</a>
                    <a href="/api/portfolio/auto-trade" class="button" onclick="return confirm('Execute automatic trading?')">Auto Trade</a>
                    <a href="/portfolio/performance" class="button">View Performance</a>
                </div>
                
                <h2>Current Holdings</h2>
                {% if holdings %}
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Quantity</th>
                            <th>Avg Cost</th>
                            <th>Current Price</th>
                            <th>Market Value</th>
                            <th>Gain/Loss</th>
                            <th>Gain/Loss %</th>
                        </tr>
                        {% for holding in holdings %}
                        <tr>
                            <td><strong>{{ holding.symbol }}</strong></td>
                            <td>{{ holding.quantity }}</td>
                            <td>${{ "%.2f"|format(holding.avg_cost) }}</td>
                            <td>${{ "%.2f"|format(holding.current_price) }}</td>
                            <td>${{ "%.2f"|format(holding.market_value) }}</td>
                            <td class="{{ 'positive' if holding.gain_loss >= 0 else 'negative' }}">
                                ${{ "%.2f"|format(holding.gain_loss) }}
                            </td>
                            <td class="{{ 'positive' if holding.gain_loss_percent >= 0 else 'negative' }}">
                                {{ "%.2f"|format(holding.gain_loss_percent) }}%
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                {% else %}
                    <p>No holdings found.</p>
                {% endif %}
                
                <h2>Recent Transactions</h2>
                {% if transactions %}
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Type</th>
                            <th>Symbol</th>
                            <th>Quantity</th>
                            <th>Price</th>
                            <th>Total</th>
                            <th>Fee</th>
                        </tr>
                        {% for tx in transactions %}
                        <tr>
                            <td>{{ tx.date }}</td>
                            <td><strong>{{ tx.type }}</strong></td>
                            <td>{{ tx.symbol }}</td>
                            <td>{{ tx.quantity }}</td>
                            <td>${{ "%.2f"|format(tx.price) }}</td>
                            <td>${{ "%.2f"|format(tx.total) }}</td>
                            <td>${{ "%.2f"|format(tx.fee) }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                {% else %}
                    <p>No recent transactions.</p>
                {% endif %}
            </div>
        </body>
        </html>
        '''
        
        return render_template_string(html_template, summary=summary, holdings=holdings, transactions=transactions)
    
    @app.route('/portfolio/trading')
    def portfolio_trading():
        """Trading interface page."""
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Trading</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
                .trading-form { background: #f9f9f9; padding: 20px; border-radius: 5px; margin-bottom: 20px; max-width: 500px; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; font-weight: bold; }
                input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 3px; }
                .button { background: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 3px; cursor: pointer; }
                .button:hover { background: #0056b3; }
                .buy-button { background: #28a745; }
                .buy-button:hover { background: #1e7e34; }
                .sell-button { background: #dc3545; }
                .sell-button:hover { background: #c82333; }
                .quote-info { background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 10px; }
                .message { padding: 10px; border-radius: 3px; margin-bottom: 15px; }
                .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
                .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            </style>
            <script>
                async function getQuote() {
                    const symbol = document.getElementById('symbol').value.toUpperCase();
                    if (!symbol) return;
                    
                    try {
                        const response = await fetch(`/api/portfolio/quote/${symbol}`);
                        const data = await response.json();
                        
                        const quoteDiv = document.getElementById('quote-info');
                        if (data.error) {
                            quoteDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                        } else {
                            quoteDiv.innerHTML = `
                                <h3>${data.symbol} - ${data.name}</h3>
                                <p><strong>Current Price:</strong> $${data.current_price ? data.current_price.toFixed(2) : 'N/A'}</p>
                                <p><strong>Sector:</strong> ${data.sector}</p>
                                <p><strong>Industry:</strong> ${data.industry}</p>
                            `;
                        }
                    } catch (error) {
                        document.getElementById('quote-info').innerHTML = `<p style="color: red;">Error fetching quote: ${error}</p>`;
                    }
                }
                
                async function executeTrade(action) {
                    const symbol = document.getElementById('symbol').value.toUpperCase();
                    const quantity = parseInt(document.getElementById('quantity').value);
                    
                    if (!symbol || !quantity || quantity <= 0) {
                        alert('Please enter valid symbol and quantity');
                        return;
                    }
                    
                    if (!confirm(`${action.toUpperCase()} ${quantity} shares of ${symbol}?`)) {
                        return;
                    }
                    
                    try {
                        const response = await fetch(`/api/portfolio/${action}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ symbol, quantity })
                        });
                        
                        const data = await response.json();
                        const messageDiv = document.getElementById('message');
                        
                        if (data.error) {
                            messageDiv.innerHTML = `<div class="message error">${data.error}</div>`;
                        } else {
                            messageDiv.innerHTML = `<div class="message success">${data.message}</div>`;
                            document.getElementById('quantity').value = '';
                        }
                    } catch (error) {
                        document.getElementById('message').innerHTML = `<div class="message error">Error: ${error}</div>`;
                    }
                }
            </script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Portfolio Trading</h1>
                    <p><a href="/portfolio">← Back to Dashboard</a> | <a href="/">Home</a></p>
                </div>
                
                <div id="message"></div>
                
                <div class="trading-form">
                    <h2>Stock Quote & Trading</h2>
                    
                    <div class="form-group">
                        <label for="symbol">Stock Symbol:</label>
                        <input type="text" id="symbol" placeholder="e.g., AAPL" style="text-transform: uppercase;">
                        <button type="button" onclick="getQuote()" class="button" style="margin-top: 10px;">Get Quote</button>
                    </div>
                    
                    <div id="quote-info" class="quote-info" style="display: none;"></div>
                    
                    <div class="form-group">
                        <label for="quantity">Quantity:</label>
                        <input type="number" id="quantity" min="1" placeholder="Number of shares">
                    </div>
                    
                    <div class="form-group">
                        <button type="button" onclick="executeTrade('buy')" class="button buy-button">BUY</button>
                        <button type="button" onclick="executeTrade('sell')" class="button sell-button">SELL</button>
                    </div>
                </div>
                
                <script>
                    // Show quote info div when symbol is entered
                    document.getElementById('symbol').addEventListener('input', function() {
                        document.getElementById('quote-info').style.display = this.value ? 'block' : 'none';
                    });
                </script>
            </div>
        </body>
        </html>
        '''
        
        return render_template_string(html_template)
    
    @app.route('/portfolio/performance')
    def portfolio_performance():
        """Portfolio performance page with charts."""
        performance_data = portfolio_service.get_portfolio_performance_data()
        
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Performance</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
                .chart-container { margin-bottom: 30px; }
                .error { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📈 Portfolio Performance</h1>
                    <p><a href="/portfolio">← Back to Dashboard</a> | <a href="/">Home</a></p>
                </div>
                
                {% if performance_data.error %}
                    <p class="error">{{ performance_data.error }}</p>
                {% else %}
                    <div class="chart-container">
                        <div id="portfolio-chart" style="width:100%;height:400px;"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div id="composition-chart" style="width:100%;height:300px;"></div>
                    </div>
                    
                    <script>
                        // Portfolio value over time
                        var portfolioTrace = {
                            x: {{ performance_data.dates | tojson }},
                            y: {{ performance_data.portfolio_values | tojson }},
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Portfolio Value',
                            line: { color: 'blue', width: 2 }
                        };
                        
                        var initialLine = {
                            x: {{ performance_data.dates | tojson }},
                            y: Array({{ performance_data.dates | length }}).fill({{ performance_data.initial_value }}),
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Initial Investment',
                            line: { color: 'gray', dash: 'dash' }
                        };
                        
                        var portfolioLayout = {
                            title: 'Portfolio Value Over Time',
                            xaxis: { title: 'Date' },
                            yaxis: { title: 'Value ($)' },
                            hovermode: 'x unified'
                        };
                        
                        Plotly.newPlot('portfolio-chart', [portfolioTrace, initialLine], portfolioLayout);
                        
                        // Portfolio composition
                        var cashTrace = {
                            x: {{ performance_data.dates | tojson }},
                            y: {{ performance_data.cash_values | tojson }},
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Cash',
                            fill: 'tonexty',
                            line: { color: 'green' }
                        };
                        
                        var holdingsTrace = {
                            x: {{ performance_data.dates | tojson }},
                            y: {{ performance_data.holdings_values | tojson }},
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Holdings',
                            line: { color: 'orange' }
                        };
                        
                        var compositionLayout = {
                            title: 'Portfolio Composition',
                            xaxis: { title: 'Date' },
                            yaxis: { title: 'Value ($)' },
                            hovermode: 'x unified'
                        };
                        
                        Plotly.newPlot('composition-chart', [cashTrace, holdingsTrace], compositionLayout);
                    </script>
                {% endif %}
            </div>
        </body>
        </html>
        '''
        
        return render_template_string(html_template, performance_data=performance_data)
    
    # API Routes
    @app.route('/api/portfolio/summary')
    def api_portfolio_summary():
        """API endpoint for portfolio summary."""
        return jsonify(portfolio_simulator.get_portfolio_summary())
    
    @app.route('/api/portfolio/holdings')
    def api_portfolio_holdings():
        """API endpoint for portfolio holdings."""
        return jsonify(portfolio_simulator._get_holdings())
    
    @app.route('/api/portfolio/quote/<symbol>')
    def api_stock_quote(symbol):
        """API endpoint for stock quotes."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return jsonify({'error': f'No data found for {symbol}'})
            
            current_price = hist['Close'].iloc[-1]
            
            return jsonify({
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'current_price': float(current_price),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            })
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/portfolio/buy', methods=['POST'])
    def api_buy_stock():
        """API endpoint for buying stocks."""
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        quantity = data.get('quantity', 0)
        
        if not symbol or quantity <= 0:
            return jsonify({'error': 'Invalid symbol or quantity'}), 400
        
        try:
            # Get current price
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return jsonify({'error': f'No price data found for {symbol}'})
            
            current_price = Decimal(str(hist['Close'].iloc[-1]))
            
            # Execute buy order
            success = portfolio_simulator._execute_buy(symbol, quantity, current_price)
            
            if success:
                return jsonify({
                    'message': f'Successfully bought {quantity} shares of {symbol} at ${current_price:.2f}',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': float(current_price)
                })
            else:
                return jsonify({'error': 'Failed to execute buy order'})
                
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/portfolio/sell', methods=['POST'])
    def api_sell_stock():
        """API endpoint for selling stocks."""
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        quantity = data.get('quantity', 0)
        
        if not symbol or quantity <= 0:
            return jsonify({'error': 'Invalid symbol or quantity'}), 400
        
        try:
            # Get current price
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return jsonify({'error': f'No price data found for {symbol}'})
            
            current_price = Decimal(str(hist['Close'].iloc[-1]))
            
            # Execute sell order
            success = portfolio_simulator._execute_sell(symbol, quantity, current_price)
            
            if success:
                return jsonify({
                    'message': f'Successfully sold {quantity} shares of {symbol} at ${current_price:.2f}',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': float(current_price)
                })
            else:
                return jsonify({'error': 'Failed to execute sell order'})
                
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/portfolio/auto-trade')
    def api_auto_trade():
        """API endpoint for automatic trading."""
        try:
            def run_auto_trade():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(portfolio_simulator.simulate_trading_day())
                loop.close()
                return result
            
            result = run_auto_trade()
            return jsonify({
                'message': f'Auto-trading completed: {result["total_trades"]} trades executed',
                'trades': result['trades_executed'],
                'portfolio_value_before': float(result['portfolio_value_before']),
                'portfolio_value_after': float(result['portfolio_value_after'])
            })
        except Exception as e:
            return jsonify({'error': str(e)})

else:
    # Portfolio not enabled - create dummy routes
    @app.route('/portfolio')
    def portfolio_disabled():
        return "Portfolio functionality not available", 404

def open_browser():
    """Open browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("🚀 Starting Stock Analyzer Web Demo")
    print("=" * 50)
    print("📊 Web interface: http://localhost:5000")
    print("🔄 The demo will automatically open in your browser")
    print("=" * 50)
    
    # Open browser in background thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

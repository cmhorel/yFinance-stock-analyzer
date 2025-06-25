"""
Unified Web Application - Clean Architecture Implementation

This web application provides:
1. Stock analysis with auto-loading of NASDAQ & TSX stocks
2. Portfolio management with centralized database
3. Interactive visualizations
4. Trading interface
"""

import asyncio
import os
import sys
import threading
import time
import webbrowser
from typing import Dict, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from flask import Flask, render_template_string, send_from_directory, jsonify, request

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.config import get_settings, setup_logging
from shared.logging import get_logger
from infrastructure.database import SqliteStockRepository, SqlitePortfolioRepository
from application.services.stock_data_service import StockDataService
from application.services.portfolio_management_service import PortfolioManagementService
from domain.services.stock_analysis_service import StockAnalysisService

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


class UnifiedWebApp:
    """Unified web application using clean architecture."""
    
    def __init__(self, db_path: str = "data/unified_analyzer.db"):
        # Setup logging
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Initialize repositories
        self.stock_repo = SqliteStockRepository(db_path=db_path)
        self.portfolio_repo = SqlitePortfolioRepository(db_path=db_path)
        
        # Initialize services
        self.stock_data_service = StockDataService(self.stock_repo)
        self.portfolio_service = PortfolioManagementService(
            self.portfolio_repo, 
            self.stock_repo, 
            self.stock_data_service
        )
        self.analysis_service = StockAnalysisService()
        
        # Create plots directory
        self.plots_path = "plots/unified_demo"
        os.makedirs(self.plots_path, exist_ok=True)
        
        # Application state
        self.stocks_loaded = False
        self.analysis_complete = False
        self.buy_candidates = []
        self.sell_candidates = []
        self.database_stats = {}
        
        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return self._render_dashboard()
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint to check application status."""
            return jsonify({
                'stocks_loaded': self.stocks_loaded,
                'analysis_complete': self.analysis_complete,
                'buy_count': len(self.buy_candidates),
                'sell_count': len(self.sell_candidates)
            })
        
        @self.app.route('/api/load-stocks', methods=['POST'])
        def api_load_stocks():
            """API endpoint to load all stocks."""
            def load_stocks():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.stock_data_service.load_all_stocks_on_startup())
                self.stocks_loaded = True
                loop.close()
                return result
            
            # Run in background thread
            thread = threading.Thread(target=load_stocks)
            thread.daemon = True
            thread.start()
            
            return jsonify({'status': 'started'})
        
        @self.app.route('/api/run-analysis', methods=['POST'])
        def api_run_analysis():
            """API endpoint to run stock analysis."""
            def run_analysis():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._run_analysis())
                loop.close()
            
            # Run analysis in background thread
            thread = threading.Thread(target=run_analysis)
            thread.daemon = True
            thread.start()
            
            return jsonify({'status': 'started'})
        
        # Portfolio routes
        @self.app.route('/portfolio')
        def portfolio_dashboard():
            """Portfolio dashboard page."""
            return self._render_portfolio_dashboard()
        
        @self.app.route('/portfolio/trading')
        def portfolio_trading():
            """Trading interface page."""
            return self._render_trading_interface()
        
        @self.app.route('/api/portfolio/summary')
        def api_portfolio_summary():
            """API endpoint for portfolio summary."""
            def get_summary():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.portfolio_service.get_portfolio_summary())
                loop.close()
                return result
            
            return jsonify(get_summary())
        
        @self.app.route('/api/portfolio/holdings')
        def api_portfolio_holdings():
            """API endpoint for portfolio holdings."""
            def get_holdings():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.portfolio_service.get_holdings_with_current_values())
                loop.close()
                return result
            
            return jsonify(get_holdings())
        
        @self.app.route('/api/portfolio/quote/<symbol>')
        def api_stock_quote(symbol):
            """API endpoint for stock quotes."""
            def get_quote():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                price = loop.run_until_complete(self.stock_data_service.get_current_price(symbol))
                loop.close()
                return price
            
            try:
                current_price = get_quote()
                if current_price is None:
                    return jsonify({'error': f'No data found for {symbol}'})
                
                return jsonify({
                    'symbol': symbol,
                    'current_price': float(current_price),
                    'name': f"{symbol} Inc."
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/portfolio/buy', methods=['POST'])
        def api_buy_stock():
            """API endpoint for buying stocks."""
            data = request.get_json()
            symbol = data.get('symbol', '').upper()
            quantity = data.get('quantity', 0)
            
            if not symbol or quantity <= 0:
                return jsonify({'error': 'Invalid symbol or quantity'}), 400
            
            def execute_buy():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.portfolio_service.execute_buy_order(symbol, quantity))
                loop.close()
                return result
            
            return jsonify(execute_buy())
        
        @self.app.route('/api/portfolio/sell', methods=['POST'])
        def api_sell_stock():
            """API endpoint for selling stocks."""
            data = request.get_json()
            symbol = data.get('symbol', '').upper()
            quantity = data.get('quantity', 0)
            
            if not symbol or quantity <= 0:
                return jsonify({'error': 'Invalid symbol or quantity'}), 400
            
            def execute_sell():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.portfolio_service.execute_sell_order(symbol, quantity))
                loop.close()
                return result
            
            return jsonify(execute_sell())
        
        # File serving routes
        @self.app.route('/plots/<path:filename>')
        def serve_plot(filename):
            """Serve plot files."""
            plots_dir = os.path.abspath('plots')
            return send_from_directory(plots_dir, filename)
        
        @self.app.route('/browse/')
        @self.app.route('/browse/<path:subpath>')
        def browse_plots(subpath=''):
            """Browse plots directory."""
            return self._render_plots_browser(subpath)
    
    async def _run_analysis(self):
        """Run the complete stock analysis."""
        self.logger.info("Starting stock analysis")
        
        # Get all stocks from database
        all_stocks = await self.stock_repo.get_all_stocks()
        if not all_stocks:
            self.logger.warning("No stocks found in database")
            return
        
        # Analyze stocks for buy/sell signals
        self.buy_candidates = []
        self.sell_candidates = []
        
        for stock in all_stocks[:20]:  # Limit for demo
            try:
                prices = await self.stock_repo.get_stock_prices(stock.id, limit=200)
                if len(prices) < 50:
                    continue
                
                # Run analysis
                analysis_result = self.analysis_service.analyze_stock(stock, prices)
                if not analysis_result:
                    continue
                
                # Extract signals
                if analysis_result.has_buy_signal:
                    self.buy_candidates.append((
                        stock.symbol,
                        analysis_result.buy_signal.score,
                        float(analysis_result.risk_score) if analysis_result.risk_score else 5.0,
                        'Unknown',  # sector
                        'Unknown',  # industry
                        float(analysis_result.buy_signal.price),
                        50.0  # rsi placeholder
                    ))
                
                if analysis_result.has_sell_signal:
                    self.sell_candidates.append((
                        stock.symbol,
                        analysis_result.sell_signal.score,
                        float(analysis_result.risk_score) if analysis_result.risk_score else 5.0,
                        'Unknown',  # sector
                        'Unknown',  # industry
                        float(analysis_result.sell_signal.price),
                        50.0  # rsi placeholder
                    ))
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {stock.symbol}: {e}")
                continue
        
        # Sort candidates
        self.buy_candidates.sort(key=lambda x: x[1], reverse=True)
        self.sell_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Create overview plot
        self._create_sector_overview_plot()
        
        # Get database stats
        sectors = await self.stock_repo.get_sectors()
        self.database_stats = {
            'total_stocks': len(all_stocks),
            'total_sectors': len(sectors),
            'sectors': sectors,
            'buy_signals': len(self.buy_candidates),
            'sell_signals': len(self.sell_candidates)
        }
        
        self.analysis_complete = True
        self.logger.info("Stock analysis completed")
    
    def _create_sector_overview_plot(self):
        """Create sector overview plot."""
        if not self.buy_candidates and not self.sell_candidates:
            return
        
        all_candidates = []
        
        for ticker, score, volatility, sector, industry, price, rsi in self.buy_candidates:
            all_candidates.append({
                'ticker': ticker, 'score': score, 'volatility': volatility,
                'sector': sector, 'industry': industry, 'recommendation': 'Buy',
                'price': price, 'rsi': rsi,
                'risk_level': 'Low' if volatility < 3.0 else 'Medium' if volatility < 6.0 else 'High'
            })
        
        for ticker, score, volatility, sector, industry, price, rsi in self.sell_candidates:
            all_candidates.append({
                'ticker': ticker, 'score': score, 'volatility': volatility,
                'sector': sector, 'industry': industry, 'recommendation': 'Sell',
                'price': price, 'rsi': rsi,
                'risk_level': 'Low' if volatility < 3.0 else 'Medium' if volatility < 6.0 else 'High'
            })
        
        if not all_candidates:
            return
        
        df_candidates = pd.DataFrame(all_candidates)
        
        fig = px.scatter(
            df_candidates, x='rsi', y='score', color='sector', symbol='recommendation',
            size='volatility', hover_data=['ticker', 'industry', 'price', 'risk_level'],
            title='Stock Recommendations: RSI vs Score (Size = Risk)',
            color_discrete_map=SECTOR_COLORS, size_max=20
        )
        
        fig.update_layout(height=700, showlegend=True)
        filename = os.path.join(self.plots_path, "sector_overview.html")
        fig.write_html(filename)
    
    def _render_dashboard(self):
        """Render the main dashboard."""
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unified Stock Analyzer</title>
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
                .btn { display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; cursor: pointer; border: none; }
                .btn:hover { background-color: #0056b3; }
                .btn.success { background-color: #28a745; }
                .btn.warning { background-color: #ffc107; color: #212529; }
                .buy { color: #28a745; font-weight: bold; }
                .sell { color: #dc3545; font-weight: bold; }
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
                            if (data.stocks_loaded && data.analysis_complete) {
                                location.reload();
                            }
                        });
                }
                
                function loadStocks() {
                    document.getElementById('load-btn').style.display = 'none';
                    document.getElementById('loading-status').style.display = 'block';
                    
                    fetch('/api/load-stocks', {method: 'POST'})
                        .then(() => {
                            // Check status every 5 seconds
                            setInterval(checkStatus, 5000);
                        });
                }
                
                function runAnalysis() {
                    document.getElementById('analysis-btn').style.display = 'none';
                    document.getElementById('analysis-status').style.display = 'block';
                    
                    fetch('/api/run-analysis', {method: 'POST'})
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
                    <h1>🚀 Unified Stock Analyzer</h1>
                    <p>Clean Architecture • Auto-Loading • Centralized Database • Portfolio Management</p>
                </div>
                
                {% if not stocks_loaded %}
                <div class="status loading" id="loading-status" style="display: none;">
                    <h3>🔄 Loading Stocks...</h3>
                    <p>Loading NASDAQ-100 and TSX stocks. This may take several minutes.</p>
                    <p><em>The page will automatically refresh when complete.</em></p>
                </div>
                
                <div style="text-align: center;">
                    <button id="load-btn" class="btn" onclick="loadStocks()">
                        📥 Load All NASDAQ & TSX Stocks
                    </button>
                    <p><em>This will automatically load all stocks on startup</em></p>
                </div>
                
                {% elif not analysis_complete %}
                <div class="status loading" id="analysis-status" style="display: none;">
                    <h3>🔄 Running Analysis...</h3>
                    <p>Analyzing stocks for buy/sell signals and generating plots.</p>
                    <p><em>The page will automatically refresh when complete.</em></p>
                </div>
                
                <div style="text-align: center;">
                    <button id="analysis-btn" class="btn success" onclick="runAnalysis()">
                        🎯 Run Stock Analysis
                    </button>
                    <p><em>Stocks loaded successfully! Ready for analysis.</em></p>
                </div>
                
                {% else %}
                
                <div class="status complete">
                    <h3>✅ Analysis Complete!</h3>
                    <p>Successfully analyzed {{ database_stats.total_stocks }} stocks</p>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-number">{{ database_stats.total_stocks }}</div>
                        <div class="stat-label">Stocks Loaded</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{{ database_stats.buy_signals }}</div>
                        <div class="stat-label">Buy Signals</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{{ database_stats.sell_signals }}</div>
                        <div class="stat-label">Sell Signals</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{{ database_stats.total_sectors }}</div>
                        <div class="stat-label">Sectors</div>
                    </div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h3>📈 Buy Recommendations</h3>
                        {% if buy_candidates %}
                        {% for ticker, score, vol, sector, industry, price, rsi in buy_candidates[:5] %}
                        <div class="buy">
                            {{ ticker }} (Score: {{ score }}) - ${{ "%.2f"|format(price) }}<br>
                            <small>Risk: {{ "%.1f"|format(vol) }}</small>
                        </div>
                        {% endfor %}
                        {% else %}
                        <p><em>No buy signals found</em></p>
                        {% endif %}
                    </div>
                    
                    <div class="card">
                        <h3>📉 Sell Recommendations</h3>
                        {% if sell_candidates %}
                        {% for ticker, score, vol, sector, industry, price, rsi in sell_candidates[:5] %}
                        <div class="sell">
                            {{ ticker }} (Score: {{ score }}) - ${{ "%.2f"|format(price) }}<br>
                            <small>Risk: {{ "%.1f"|format(vol) }}</small>
                        </div>
                        {% endfor %}
                        {% else %}
                        <p><em>No sell signals found</em></p>
                        {% endif %}
                    </div>
                </div>
                
                <div class="card">
                    <h3>📊 Navigation</h3>
                    <a href="/plots/unified_demo/sector_overview.html" target="_blank" class="btn success">
                        🎯 View Analysis Charts
                    </a>
                    <a href="/portfolio" class="btn warning">
                        💼 Portfolio Management
                    </a>
                    <a href="/browse" target="_blank" class="btn">
                        📁 Browse All Files
                    </a>
                </div>
                
                {% endif %}
            </div>
        </body>
        </html>
        '''
        
        return render_template_string(
            html_template,
            stocks_loaded=self.stocks_loaded,
            analysis_complete=self.analysis_complete,
            buy_candidates=self.buy_candidates,
            sell_candidates=self.sell_candidates,
            database_stats=self.database_stats
        )
    
    def _render_portfolio_dashboard(self):
        """Render portfolio dashboard."""
        # Get portfolio data synchronously for template
        def get_portfolio_data():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            summary = loop.run_until_complete(self.portfolio_service.get_portfolio_summary())
            holdings = loop.run_until_complete(self.portfolio_service.get_holdings_with_current_values())
            transactions = loop.run_until_complete(self.portfolio_service.get_recent_transactions(10))
            loop.close()
            return summary, holdings, transactions
        
        summary, holdings, transactions = get_portfolio_data()
        
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
                .btn { background: #007bff; color: white; padding: 10px 15px; text-decoration: none; border-radius: 3px; margin-right: 10px; }
                .btn:hover { background: #0056b3; }
                .error { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📈 Portfolio Dashboard</h1>
                    <p><a href="/">← Back to Home</a> | <a href="/portfolio/trading">Trading</a></p>
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
                        <p><strong>Can Trade:</strong> {{ "Yes" if summary.can_trade else "No (cooldown active)" }}</p>
                    {% endif %}
                </div>
                
                <div style="margin-bottom: 20px;">
                    <a href="/portfolio/trading" class="btn">Manual Trading</a>
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
    
    def _render_trading_interface(self):
        """Render trading interface."""
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
                        const response = await fetch(`/api/portfolio/quote/${symbol}`

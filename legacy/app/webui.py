#!/home/chorel/venv/bin/python
from flask import Flask, send_from_directory, render_template_string, request, jsonify
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

app = Flask(__name__)
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))

# Import portfolio service
try:
    from domain.services.portfolio_service import portfolio_service
    PORTFOLIO_ENABLED = True
    print("Portfolio service loaded successfully!")
except ImportError as e:
    print(f"Portfolio service not available: {e}")
    PORTFOLIO_ENABLED = False
except Exception as e:
    print(f"Error loading portfolio service: {e}")
    PORTFOLIO_ENABLED = False

print(f"PORTFOLIO_ENABLED: {PORTFOLIO_ENABLED}")

def get_directory_contents(path):
    """Get files and directories in the given path."""
    full_path = os.path.join(PLOTS_DIR, path) if path else PLOTS_DIR
    if not os.path.exists(full_path):
        return [], []
    
    items = os.listdir(full_path)
    directories = [item for item in items if os.path.isdir(os.path.join(full_path, item))]
    files = [item for item in items if item.endswith('.html')]
    
    return sorted(directories), sorted(files)

@app.route('/')
@app.route('/browse/')
@app.route('/browse/<path:subpath>')
def list_plots(subpath=''):
    directories, files = get_directory_contents(subpath)
    
    # Build breadcrumb navigation
    breadcrumbs = []
    if subpath:
        parts = subpath.split('/')
        current_path = ''
        for part in parts:
            current_path = os.path.join(current_path, part).replace('\\', '/')
            breadcrumbs.append(f'<a href="/browse/{current_path}">{part}</a>')
    
    breadcrumb_html = ' / '.join(['<a href="/browse/">Home</a>'] + breadcrumbs)
    
    # Build directory links
    dir_links = []
    for dirname in directories:
        dir_path = os.path.join(subpath, dirname).replace('\\', '/') if subpath else dirname
        dir_links.append(f'<li>📁 <a href="/browse/{dir_path}"><strong>{dirname}/</strong></a></li>')
    
    # Build file links
    file_links = []
    for filename in files:
        file_path = os.path.join(subpath, filename).replace('\\', '/') if subpath else filename
        file_links.append(f'<li>📊 <a href="/plots/{file_path}" target="_blank">{filename}</a></li>')
    
    # Combine all links
    all_links = dir_links + file_links
    
    # Add back navigation if in subdirectory
    back_link = ''
    if subpath:
        parent_path = '/'.join(subpath.split('/')[:-1]) if '/' in subpath else ''
        back_link = f'<p><a href="/browse/{parent_path}">⬅️ Back to parent directory</a></p>'
    
    current_dir = f" - {subpath}" if subpath else ""
    
    # Add portfolio navigation if enabled
    portfolio_nav = ''
    if PORTFOLIO_ENABLED:
        portfolio_nav = '''
        <div style="background-color: #f0f8ff; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h3>📈 Portfolio Management</h3>
            <p><a href="/portfolio" style="margin-right: 15px;">Portfolio Dashboard</a> | 
               <a href="/portfolio/trading" style="margin-right: 15px;">Trading</a> | 
               <a href="/portfolio/performance">Performance</a></p>
        </div>
        '''
    
    html_template = '''
    <h1>Stock Analysis System{{current_dir}}</h1>
    <p><strong>Navigation:</strong> {{breadcrumbs|safe}}</p>
    {{portfolio_nav|safe}}
    {{back_link|safe}}
    <h2>📊 Analysis Plots</h2>
    <ul style="list-style-type: none; padding-left: 0;">
        {{links|safe}}
    </ul>
    <hr>
    <p><em>📁 = Folder (click to browse) | 📊 = Plot file (click to view)</em></p>
    '''
    
    return render_template_string(
        html_template,
        current_dir=current_dir,
        breadcrumbs=breadcrumb_html,
        portfolio_nav=portfolio_nav,
        back_link=back_link,
        links=''.join(all_links) if all_links else '<li><em>No plots or folders found</em></li>'
    )

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

# Portfolio Management Routes
if PORTFOLIO_ENABLED:
    
    @app.route('/portfolio')
    def portfolio_dashboard():
        """Portfolio dashboard page."""
        summary = portfolio_service.get_portfolio_summary()
        holdings = portfolio_service.get_holdings()
        transactions = portfolio_service.get_recent_transactions(10)
        
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
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
            <h1>📈 Portfolio Dashboard</h1>
            <p><a href="/">← Back to Home</a> | <a href="/portfolio/trading">Trading</a> | <a href="/portfolio/performance">Performance</a></p>
            
            <div class="summary">
                <h2>Portfolio Summary</h2>
                {% if summary.error %}
                    <p class="error">{{ summary.error }}</p>
                {% else %}
                    <p><strong>Cash Balance:</strong> ${{ "%.2f"|format(summary.cash_balance) }}</p>
                    <p><strong>Total Portfolio Value:</strong> ${{ "%.2f"|format(summary.total_value) }}</p>
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
                body { font-family: Arial, sans-serif; margin: 20px; }
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
            <h1>📊 Portfolio Trading</h1>
            <p><a href="/portfolio">← Back to Dashboard</a> | <a href="/">Home</a></p>
            
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
                body { font-family: Arial, sans-serif; margin: 20px; }
                .chart-container { margin-bottom: 30px; }
                .error { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>📈 Portfolio Performance</h1>
            <p><a href="/portfolio">← Back to Dashboard</a> | <a href="/">Home</a></p>
            
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
        </body>
        </html>
        '''
        
        return render_template_string(html_template, performance_data=performance_data)
    
    # API Routes
    @app.route('/api/portfolio/summary')
    def api_portfolio_summary():
        """API endpoint for portfolio summary."""
        return jsonify(portfolio_service.get_portfolio_summary())
    
    @app.route('/api/portfolio/holdings')
    def api_portfolio_holdings():
        """API endpoint for portfolio holdings."""
        return jsonify(portfolio_service.get_holdings())
    
    @app.route('/api/portfolio/transactions')
    def api_portfolio_transactions():
        """API endpoint for portfolio transactions."""
        limit = request.args.get('limit', 20, type=int)
        return jsonify(portfolio_service.get_recent_transactions(limit))
    
    @app.route('/api/portfolio/quote/<symbol>')
    def api_stock_quote(symbol):
        """API endpoint for stock quotes."""
        return jsonify(portfolio_service.get_stock_quote(symbol))
    
    @app.route('/api/portfolio/buy', methods=['POST'])
    def api_buy_stock():
        """API endpoint for buying stocks."""
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        quantity = data.get('quantity', 0)
        
        if not symbol or quantity <= 0:
            return jsonify({'error': 'Invalid symbol or quantity'}), 400
        
        result = portfolio_service.execute_buy_order(symbol, quantity)
        return jsonify(result)
    
    @app.route('/api/portfolio/sell', methods=['POST'])
    def api_sell_stock():
        """API endpoint for selling stocks."""
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        quantity = data.get('quantity', 0)
        
        if not symbol or quantity <= 0:
            return jsonify({'error': 'Invalid symbol or quantity'}), 400
        
        result = portfolio_service.execute_sell_order(symbol, quantity)
        return jsonify(result)
    
    @app.route('/api/portfolio/auto-trade')
    def api_auto_trade():
        """API endpoint for automatic trading."""
        result = portfolio_service.execute_auto_trading()
        return jsonify(result)
    
    @app.route('/api/portfolio/performance')
    def api_portfolio_performance():
        """API endpoint for portfolio performance data."""
        return jsonify(portfolio_service.get_portfolio_performance_data())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

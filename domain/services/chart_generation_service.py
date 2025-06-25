"""Chart generation service for creating all types of analysis charts."""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.logging import get_logger
from domain.entities.stock import StockPrice
from domain.services.technical_analysis_service import TechnicalAnalysisService


class ChartGenerationService:
    """Service for generating all types of analysis charts."""
    
    def __init__(self, stock_repo, portfolio_repo=None):
        self.stock_repo = stock_repo
        self.portfolio_repo = portfolio_repo
        self.technical_analysis = TechnicalAnalysisService()
        self.logger = get_logger(__name__)
        
        # Define sector color mapping
        self.sector_colors = {
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
        
        # Ensure plots directory exists
        self.plots_dir = "plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, "stock_analysis"), exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, "web_demo"), exist_ok=True)
    
    def get_sector_color(self, sector: str) -> str:
        """Get color for a given sector."""
        return self.sector_colors.get(sector, self.sector_colors['Unknown'])
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI for a series of prices."""
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        series = pd.Series(prices)
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        
        return rsi.tolist()
    
    def calculate_volatility(self, prices: List[float], period: int = 20) -> List[float]:
        """Calculate rolling volatility."""
        if len(prices) < period:
            return [0.2] * len(prices)
        
        series = pd.Series(prices)
        returns = series.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)
        volatility = volatility.fillna(0.2)
        
        return volatility.tolist()
    
    def generate_buy_sell_signals(self, prices: List[StockPrice]) -> Tuple[List[int], List[int]]:
        """Generate buy/sell signals for visualization."""
        if len(prices) < 50:
            return [0] * len(prices), [0] * len(prices)
        
        close_prices = [float(p.close_price) for p in prices]
        volumes = [float(p.volume) for p in prices]
        
        # Calculate indicators
        ma20 = pd.Series(close_prices).rolling(window=20).mean().tolist()
        ma50 = pd.Series(close_prices).rolling(window=50).mean().tolist()
        rsi = self.calculate_rsi(close_prices)
        volatility = self.calculate_volatility(close_prices)
        
        buy_signals = [0] * len(prices)
        sell_signals = [0] * len(prices)
        
        for i in range(50, len(prices)):
            current_close = close_prices[i]
            current_ma20 = ma20[i]
            current_ma50 = ma50[i]
            current_rsi = rsi[i]
            current_vol = volatility[i]
            
            # 7-day momentum
            momentum_7d = (close_prices[i] - close_prices[i-7]) / close_prices[i-7] if i >= 7 else 0
            
            # Volume surge
            avg_volume = np.mean(volumes[max(0, i-20):i]) if i >= 20 else np.mean(volumes[:i+1])
            volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1
            
            # Buy signal conditions
            buy_conditions = [
                current_close > current_ma20,
                current_ma20 > current_ma50,
                current_rsi < 40,
                momentum_7d > 0.02,
                volume_ratio > 1.3,
                current_vol < 0.4
            ]
            
            if sum(buy_conditions) >= 4:
                buy_signals[i] = 1
            
            # Sell signal conditions
            sell_conditions = [
                current_close < current_ma20,
                current_ma20 < current_ma50,
                current_rsi > 70,
                momentum_7d < -0.02,
                current_vol > 0.5
            ]
            
            if sum(sell_conditions) >= 3:
                sell_signals[i] = 1
        
        return buy_signals, sell_signals
    
    async def generate_stock_analysis_chart(self, symbol: str) -> bool:
        """Generate detailed stock analysis chart for a symbol."""
        try:
            # Get stock data
            stock = await self.stock_repo.get_stock_by_symbol(symbol)
            if not stock:
                self.logger.warning(f"Stock {symbol} not found")
                return False
            
            prices = await self.stock_repo.get_stock_prices(stock.id, limit=200)
            if len(prices) < 20:
                self.logger.warning(f"Insufficient data for {symbol}")
                return False
            
            # Convert to lists for calculations
            dates = [p.date for p in prices]
            close_prices = [float(p.close_price) for p in prices]
            high_prices = [float(p.high_price) for p in prices]
            low_prices = [float(p.low_price) for p in prices]
            volumes = [float(p.volume) for p in prices]
            
            # Calculate indicators
            ma20 = pd.Series(close_prices).rolling(window=20).mean().tolist()
            ma50 = pd.Series(close_prices).rolling(window=50).mean().tolist()
            rsi = self.calculate_rsi(close_prices)
            volatility = self.calculate_volatility(close_prices)
            
            # Generate signals
            buy_signals, sell_signals = self.generate_buy_sell_signals(prices)
            
            # Get sector info
            sector = getattr(stock, 'sector', 'Unknown')
            industry = getattr(stock, 'industry', 'Unknown')
            sector_color = self.get_sector_color(sector)
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                subplot_titles=(f'{symbol} - {sector} ({industry})', 'RSI', 'Volatility', 'Volume'),
                row_heights=[0.5, 0.15, 0.15, 0.2]
            )
            
            # Price and Moving Averages
            fig.add_trace(
                go.Scatter(
                    x=dates, 
                    y=close_prices,
                    mode='lines',
                    name='Close Price',
                    line=dict(color=sector_color, width=2),
                    hovertemplate='<b>Close Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=dates, 
                    y=ma20,
                    mode='lines',
                    name='20-day MA',
                    line=dict(color='orange', width=1),
                    hovertemplate='<b>20-day MA</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=dates, 
                    y=ma50,
                    mode='lines',
                    name='50-day MA',
                    line=dict(color='green', width=1),
                    hovertemplate='<b>50-day MA</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add buy/sell signals
            buy_dates = [dates[i] for i, signal in enumerate(buy_signals) if signal == 1]
            buy_prices = [close_prices[i] for i, signal in enumerate(buy_signals) if signal == 1]
            sell_dates = [dates[i] for i, signal in enumerate(sell_signals) if signal == 1]
            sell_prices = [close_prices[i] for i, signal in enumerate(sell_signals) if signal == 1]
            
            if buy_dates:
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates, 
                        y=buy_prices,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        hovertemplate='<b>Buy Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if sell_dates:
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates, 
                        y=sell_prices,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        hovertemplate='<b>Sell Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # RSI
            fig.add_trace(
                go.Scatter(
                    x=dates, 
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1),
                    hovertemplate='<b>RSI</b><br>Date: %{x}<br>RSI: %{y:.1f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # RSI reference lines
            fig.add_shape(
                type="line", x0=dates[0], x1=dates[-1],
                y0=70, y1=70, line=dict(color="red", width=1, dash="dash"),
                row=2, col=1
            )
            fig.add_shape(
                type="line", x0=dates[0], x1=dates[-1],
                y0=30, y1=30, line=dict(color="green", width=1, dash="dash"),
                row=2, col=1
            )
            
            # Volatility
            fig.add_trace(
                go.Scatter(
                    x=dates, 
                    y=volatility,
                    mode='lines',
                    name='Volatility',
                    line=dict(color='red', width=1),
                    hovertemplate='<b>Volatility</b><br>Date: %{x}<br>Volatility: %{y:.3f}<extra></extra>'
                ),
                row=3, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=dates, 
                    y=volumes,
                    name='Volume',
                    marker_color='lightgray',
                    hovertemplate='<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
                ),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Stock Analysis - {sector} Sector',
                xaxis_title='Date',
                height=900,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig.update_yaxes(title_text="Volatility", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)
            
            # Save chart
            filename = os.path.join(self.plots_dir, "stock_analysis", f"{symbol}_analysis.html")
            fig.write_html(filename)
            
            # Also save to web_demo directory for compatibility
            web_demo_filename = os.path.join(self.plots_dir, "web_demo", f"{symbol}_analysis.html")
            fig.write_html(web_demo_filename)
            
            self.logger.info(f"Generated chart for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating chart for {symbol}: {e}")
            return False
    
    async def generate_sector_overview_chart(self, recommendations: List[Dict[str, Any]]) -> bool:
        """Generate sector overview chart with recommendations."""
        try:
            if not recommendations:
                self.logger.warning("No recommendations provided for sector overview")
                return False
            
            # Prepare data for visualization
            chart_data = []
            for rec in recommendations:
                chart_data.append({
                    'symbol': rec['symbol'],
                    'sentiment': rec.get('sentiment', 0),
                    'score': rec.get('confidence', rec.get('score', 50)),
                    'sector': rec.get('sector', 'Unknown'),
                    'industry': rec.get('industry', 'Unknown'),
                    'recommendation': rec['recommendation'],
                    'volatility': rec.get('volatility', 0.2),
                    'risk_level': 'Low' if rec.get('volatility', 0.2) < 0.3 else 'Medium' if rec.get('volatility', 0.2) < 0.5 else 'High'
                })
            
            if not chart_data:
                return False
            
            df_candidates = pd.DataFrame(chart_data)
            
            # Create scatter plot
            fig = px.scatter(
                df_candidates,
                x='sentiment',
                y='score',
                color='sector',
                symbol='recommendation',
                size='volatility',
                hover_data=['symbol', 'industry', 'volatility', 'risk_level'],
                title='Stock Recommendations: Sentiment vs Score (Size = Volatility)',
                labels={
                    'sentiment': 'Average Sentiment Score',
                    'score': 'Recommendation Score',
                    'sector': 'Sector',
                    'volatility': 'Volatility'
                },
                color_discrete_map=self.sector_colors,
                size_max=20
            )
            
            # Add quadrant lines
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add quadrant annotations
            fig.add_annotation(x=0.3, y=80, text="High Score<br>Positive Sentiment", 
                              showarrow=False, bgcolor="lightgreen", opacity=0.7)
            fig.add_annotation(x=-0.3, y=80, text="High Score<br>Negative Sentiment", 
                              showarrow=False, bgcolor="lightyellow", opacity=0.7)
            fig.add_annotation(x=0.3, y=30, text="Low Score<br>Positive Sentiment", 
                              showarrow=False, bgcolor="lightblue", opacity=0.7)
            fig.add_annotation(x=-0.3, y=30, text="Low Score<br>Negative Sentiment", 
                              showarrow=False, bgcolor="lightcoral", opacity=0.7)
            
            fig.update_layout(
                height=700,
                showlegend=True,
                xaxis_title="News Sentiment Score (Negative ← → Positive)",
                yaxis_title="Recommendation Score (Higher = Stronger Signal)"
            )
            
            # Save chart
            filename = os.path.join(self.plots_dir, "sector_overview.html")
            fig.write_html(filename)
            
            # Also create volatility risk chart
            await self.generate_volatility_risk_chart(df_candidates)
            
            self.logger.info("Generated sector overview chart")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating sector overview chart: {e}")
            return False
    
    async def generate_volatility_risk_chart(self, df_candidates: pd.DataFrame) -> bool:
        """Generate volatility risk analysis chart."""
        try:
            # Create a copy and fix sentiment for size (must be positive)
            df_viz = df_candidates.copy()
            df_viz['sentiment_abs'] = abs(df_viz['sentiment']) + 0.1
            
            # Create volatility vs score chart
            fig = px.scatter(
                df_viz,
                x='volatility',
                y='score',
                color='recommendation',
                size='sentiment_abs',
                hover_data=['symbol', 'sector', 'industry', 'risk_level', 'sentiment'],
                title='Risk Analysis: Volatility vs Recommendation Score',
                labels={
                    'volatility': 'Annualized Volatility',
                    'score': 'Recommendation Score',
                    'sentiment_abs': 'Sentiment Strength'
                },
                color_discrete_map={'Buy': 'green', 'Sell': 'red', 'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
            )
            
            # Add risk zones
            fig.add_vrect(x0=0, x1=0.3, fillcolor="green", opacity=0.1, 
                          annotation_text="Low Risk", annotation_position="top left")
            fig.add_vrect(x0=0.3, x1=0.5, fillcolor="yellow", opacity=0.1, 
                          annotation_text="Medium Risk", annotation_position="top")
            fig.add_vrect(x0=0.5, x1=2, fillcolor="red", opacity=0.1, 
                          annotation_text="High Risk", annotation_position="top right")
            
            fig.update_layout(
                height=600,
                showlegend=True,
                xaxis_title="Volatility (Risk Level)",
                yaxis_title="Recommendation Score"
            )
            
            # Save chart
            filename = os.path.join(self.plots_dir, "volatility_risk_analysis.html")
            fig.write_html(filename)
            
            self.logger.info("Generated volatility risk analysis chart")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating volatility risk chart: {e}")
            return False
    
    async def generate_portfolio_performance_chart(self, portfolio_service) -> bool:
        """Generate portfolio performance chart using actual portfolio history."""
        try:
            if not self.portfolio_repo:
                self.logger.warning("No portfolio repository available")
                return False
            
            # Get portfolio data
            portfolios = await self.portfolio_repo.get_all_portfolios()
            if not portfolios:
                self.logger.warning("No portfolios found")
                return False
            
            portfolio = portfolios[0]
            
            # Get actual portfolio history from database
            history = await self.portfolio_repo.get_portfolio_history(portfolio.id, days=90)
            
            if not history:
                # If no history exists, create initial snapshot and show current value
                current_summary = await portfolio_service.get_portfolio_summary()
                if 'total_value' in current_summary:
                    # Record current snapshot
                    await portfolio_service._record_portfolio_snapshot(portfolio.id)
                    # Try to get history again
                    history = await self.portfolio_repo.get_portfolio_history(portfolio.id, days=90)
                
                if not history:
                    # Still no history, create a simple chart with current value
                    dates = [datetime.now().date()]
                    values = [current_summary.get('total_value', 100000)]
                    cash_values = [current_summary.get('cash_balance', 100000)]
                    holdings_values = [current_summary.get('total_value', 100000) - current_summary.get('cash_balance', 0)]
                else:
                    dates = [datetime.fromisoformat(h['date']).date() if isinstance(h['date'], str) else h['date'] for h in history]
                    values = [h['total_value'] for h in history]
                    cash_values = [h['cash_balance'] for h in history]
                    holdings_values = [h['holdings_value'] for h in history]
            else:
                # Use actual historical data
                dates = [datetime.fromisoformat(h['date']).date() if isinstance(h['date'], str) else h['date'] for h in history]
                values = [h['total_value'] for h in history]
                cash_values = [h['cash_balance'] for h in history]
                holdings_values = [h['holdings_value'] for h in history]
            
            # Create chart with multiple traces
            fig = go.Figure()
            
            # Total portfolio value
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name='Total Portfolio Value',
                line=dict(color='blue', width=3),
                hovertemplate='<b>Total Value</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
            
            # Cash balance
            fig.add_trace(go.Scatter(
                x=dates,
                y=cash_values,
                mode='lines',
                name='Cash Balance',
                line=dict(color='green', width=2, dash='dot'),
                hovertemplate='<b>Cash</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
            
            # Holdings value
            fig.add_trace(go.Scatter(
                x=dates,
                y=holdings_values,
                mode='lines',
                name='Holdings Value',
                line=dict(color='orange', width=2, dash='dash'),
                hovertemplate='<b>Holdings</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
            
            # Add benchmark line (initial investment)
            initial_value = 100000
            fig.add_hline(y=initial_value, line_dash="dash", line_color="gray", 
                         annotation_text="Initial Investment ($100,000)")
            
            # Calculate performance metrics
            if len(values) > 0:
                current_value = values[-1]
                total_return = current_value - initial_value
                return_pct = (total_return / initial_value) * 100
                
                # Calculate daily returns if we have multiple data points
                daily_returns = []
                if len(values) > 1:
                    for i in range(1, len(values)):
                        if values[i-1] != 0:
                            daily_return = (values[i] - values[i-1]) / values[i-1]
                            daily_returns.append(daily_return)
                
                # Calculate volatility (annualized)
                volatility = 0
                if len(daily_returns) > 1:
                    volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized volatility in %
                
                # Calculate max drawdown
                max_drawdown = 0
                peak = values[0]
                for value in values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                max_drawdown_pct = max_drawdown * 100
                
                title = f'Portfolio Performance - Total Return: ${total_return:,.2f} ({return_pct:+.2f}%)'
                if volatility > 0:
                    title += f' | Volatility: {volatility:.1f}% | Max Drawdown: {max_drawdown_pct:.1f}%'
            else:
                title = 'Portfolio Performance'
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                height=600,
                showlegend=True,
                hovermode='x unified',
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )
            
            # Add annotations for key metrics
            if len(values) > 0:
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=f"Current Value: ${values[-1]:,.2f}<br>" +
                         f"Cash: ${cash_values[-1]:,.2f}<br>" +
                         f"Holdings: ${holdings_values[-1]:,.2f}<br>" +
                         f"Data Points: {len(values)}",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    font=dict(size=10)
                )
            
            # Save chart
            filename = os.path.join(self.plots_dir, "portfolio_performance.html")
            fig.write_html(filename)
            
            self.logger.info(f"Generated portfolio performance chart with {len(values)} data points")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio performance chart: {e}")
            return False
    
    async def generate_all_charts(self, portfolio_service=None, recommendations=None) -> Dict[str, Any]:
        """Generate all charts and return status."""
        self.logger.info("Starting comprehensive chart generation...")
        
        results = {
            'stock_charts': {'generated': 0, 'failed': 0, 'symbols': []},
            'sector_overview': False,
            'volatility_risk': False,
            'portfolio_performance': False,
            'total_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            # Get all stocks
            stocks = await self.stock_repo.get_all_stocks()
            self.logger.info(f"Generating charts for {len(stocks)} stocks...")
            
            # Generate individual stock charts
            for stock in stocks:
                try:
                    success = await self.generate_stock_analysis_chart(stock.symbol)
                    if success:
                        results['stock_charts']['generated'] += 1
                        results['stock_charts']['symbols'].append(stock.symbol)
                    else:
                        results['stock_charts']['failed'] += 1
                except Exception as e:
                    self.logger.error(f"Error generating chart for {stock.symbol}: {e}")
                    results['stock_charts']['failed'] += 1
            
            # Generate sector overview chart
            if recommendations:
                results['sector_overview'] = await self.generate_sector_overview_chart(recommendations)
            
            # Generate portfolio performance chart
            if portfolio_service:
                results['portfolio_performance'] = await self.generate_portfolio_performance_chart(portfolio_service)
            
            # Calculate total time
            end_time = datetime.now()
            results['total_time'] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Chart generation completed in {results['total_time']:.2f} seconds")
            self.logger.info(f"Stock charts: {results['stock_charts']['generated']} generated, {results['stock_charts']['failed']} failed")
            self.logger.info(f"Sector overview: {'✓' if results['sector_overview'] else '✗'}")
            self.logger.info(f"Portfolio performance: {'✓' if results['portfolio_performance'] else '✗'}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive chart generation: {e}")
            results['error'] = str(e)
            return results

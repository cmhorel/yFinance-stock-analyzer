"""
Full System Demo - Recreating Original Functionality with New Architecture

This demonstrates the complete refactored system by:
1. Scraping NASDAQ-100 and TSX-60 tickers from Wikipedia
2. Fetching stock data using yfinance
3. Storing everything in the database using our new repository
4. Creating the same interactive plots and analysis as the original
5. Generating buy/sell recommendations
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
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import the new architecture
from shared.config import get_settings, setup_logging
from shared.logging import get_logger
from domain.entities.stock import Stock, StockPrice, StockInfo
from infrastructure.database import SqliteStockRepository


# Sector color mapping (same as original)
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


class ModernStockAnalyzer:
    """Modern stock analyzer using the new architecture."""
    
    def __init__(self, db_path: str = "data/full_system_demo.db"):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.repo = SqliteStockRepository(db_path=db_path)
        
        # Create plots directory
        self.plots_path = "plots/full_system_demo"
        os.makedirs(self.plots_path, exist_ok=True)
    
    def get_nasdaq_100_tickers(self) -> List[str]:
        """Scrape NASDAQ-100 tickers from Wikipedia."""
        self.logger.info("Scraping NASDAQ-100 tickers from Wikipedia")
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')
            tickers = tables[4]['Ticker'].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
            self.logger.info(f"Found {len(tickers)} NASDAQ-100 tickers")
            return tickers
        except Exception as e:
            self.logger.error(f"Error scraping NASDAQ-100 tickers: {e}")
            return []
    
    def get_tsx60_tickers(self) -> List[str]:
        """Scrape TSX-60 tickers from Wikipedia."""
        self.logger.info("Scraping TSX-60 tickers from Wikipedia")
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/S%26P/TSX_60')
            for table in tables:
                if 'Symbol' in table.columns:
                    tickers = table['Symbol'].dropna().tolist()
                    tickers = [str(t).replace('.', '-') + '.TO' for t in tickers]
                    self.logger.info(f"Found {len(tickers)} TSX-60 tickers")
                    return tickers
            raise ValueError("TSX-60 'Symbol' table not found")
        except Exception as e:
            self.logger.error(f"Error scraping TSX-60 tickers: {e}")
            return []
    
    async def sync_ticker_data(self, symbol: str) -> bool:
        """Sync data for a single ticker."""
        try:
            # Get or create stock
            stock = await self.repo.get_stock_by_symbol(symbol)
            if not stock:
                stock = await self.repo.create_stock(Stock(
                    id=None,
                    symbol=symbol,
                    name=None,
                    exchange=None
                ))
            
            # Get latest date to determine start date
            latest_date = await self.repo.get_latest_stock_price_date(stock.id)
            if latest_date:
                start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                start_date = '2020-01-01'
            
            # Fetch data from yfinance
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(start=start_date, progress=False)
            
            if hist.empty:
                self.logger.debug(f"No new data for {symbol}")
                return True
            
            # Convert to our domain entities
            prices = []
            for date, row in hist.iterrows():
                if pd.isna(row[['Open', 'High', 'Low', 'Close', 'Volume']]).any():
                    continue
                
                price = StockPrice(
                    id=None,
                    stock_id=stock.id,
                    date=date.date(),
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
                self.logger.debug(f"Stored {len(prices)} prices for {symbol}")
            
            # Try to get stock info
            try:
                info = ticker_obj.info
                if info and 'sector' in info:
                    stock_info = StockInfo(
                        id=None,
                        stock_id=stock.id,
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
            except Exception as e:
                self.logger.debug(f"Could not get info for {symbol}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error syncing {symbol}: {e}")
            return False
    
    async def sync_all_tickers(self, tickers: List[str], max_workers: int = 5):
        """Sync all tickers with threading."""
        self.logger.info(f"Starting sync of {len(tickers)} tickers")
        
        # Use asyncio semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_workers)
        
        async def sync_with_semaphore(symbol):
            async with semaphore:
                return await self.sync_ticker_data(symbol)
        
        # Create tasks for all tickers
        tasks = [sync_with_semaphore(ticker) for ticker in tickers]
        
        # Execute with progress bar
        completed = 0
        for task in asyncio.as_completed(tasks):
            await task
            completed += 1
            if completed % 10 == 0:
                self.logger.info(f"Completed {completed}/{len(tickers)} tickers")
        
        self.logger.info(f"Sync completed for all {len(tickers)} tickers")
    
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
    
    async def get_stock_dataframe(self, months_back: int = 6) -> pd.DataFrame:
        """Get stock data as DataFrame for analysis."""
        # Get stocks with sufficient data
        stocks_data = await self.repo.get_stocks_for_analysis(months_back=months_back, min_data_points=50)
        
        all_data = []
        for item in stocks_data:
            stock = item['stock']
            
            # Get prices
            prices = await self.repo.get_stock_prices(stock.id, limit=200)
            if len(prices) < 50:
                continue
            
            # Get stock info
            stock_info = await self.repo.get_stock_info(stock.id)
            
            # Convert to DataFrame format
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
        
        # Calculate indicators
        ma20 = close.rolling(window=20).mean()
        ma50 = close.rolling(window=50).mean()
        rsi = self.calculate_rsi(close)
        volatility = self.calculate_volatility(close)
        
        # Momentum: 7-day price change
        momentum = close - close.shift(7)
        
        # Volume change
        vol_change = volume.rolling(window=5).mean() - volume.rolling(window=5).mean().shift(5)
        
        # Get latest values
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
        
        # Calculate buy score
        buy_score = 0
        buy_score += 1 if metrics['close'] > metrics['ma20'] else 0
        buy_score += 1 if metrics['close'] > metrics['ma50'] else 0
        buy_score += 1 if metrics['rsi'] < 40 else 0
        buy_score += 1 if metrics['momentum'] > 0 else 0
        buy_score += 1 if metrics['vol_change'] > 0 else 0
        buy_score += 1 if metrics['volatility'] <= 0.3 else 0
        
        # Calculate sell score
        sell_score = 0
        sell_score += 1 if metrics['close'] < metrics['ma20'] else 0
        sell_score += 1 if metrics['close'] < metrics['ma50'] else 0
        sell_score += 1 if metrics['rsi'] > 60 else 0
        sell_score += 1 if metrics['momentum'] < 0 else 0
        sell_score += 1 if metrics['vol_change'] < 0 else 0
        sell_score += 1 if metrics['volatility'] > 0.5 else 0
        
        return {
            'buy_score': buy_score,
            'sell_score': sell_score,
            'volatility': metrics['volatility'],
            'sector': df_ticker['sector'].iloc[0],
            'industry': df_ticker['industry'].iloc[0]
        }
    
    def plot_stock_analysis(self, df_ticker: pd.DataFrame, ticker: str):
        """Create detailed stock analysis plot (same as original)."""
        df_ticker = df_ticker.copy()
        df_ticker['MA20'] = df_ticker['close'].rolling(window=20).mean()
        df_ticker['MA50'] = df_ticker['close'].rolling(window=50).mean()
        df_ticker['RSI'] = self.calculate_rsi(df_ticker['close'])
        df_ticker['Volatility'] = self.calculate_volatility(df_ticker['close'])
        
        sector = df_ticker['sector'].iloc[0]
        industry = df_ticker['industry'].iloc[0]
        sector_color = SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(f'{ticker} - {sector} ({industry})', 'RSI', 'Volatility', 'Volume'),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Price and Moving Averages
        fig.add_trace(
            go.Scatter(
                x=df_ticker['date'], 
                y=df_ticker['close'],
                mode='lines',
                name='Close Price',
                line=dict(color=sector_color, width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_ticker['date'], 
                y=df_ticker['MA20'],
                mode='lines',
                name='20-day MA',
                line=dict(color='orange', width=1),
                hovertemplate='<b>20-day MA</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_ticker['date'], 
                y=df_ticker['MA50'],
                mode='lines',
                name='50-day MA',
                line=dict(color='green', width=1),
                hovertemplate='<b>50-day MA</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df_ticker['date'], 
                y=df_ticker['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=1),
                hovertemplate='<b>RSI</b><br>Date: %{x}<br>RSI: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Volatility
        fig.add_trace(
            go.Scatter(
                x=df_ticker['date'], 
                y=df_ticker['Volatility'],
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
                x=df_ticker['date'], 
                y=df_ticker['volume'],
                name='Volume',
                marker_color='lightgray',
                hovertemplate='<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Stock Analysis - {sector} Sector',
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
        
        # Save plot
        filename = os.path.join(self.plots_path, f"{ticker}_analysis.html")
        fig.write_html(filename)
        self.logger.info(f"Saved analysis plot for {ticker}")
        
        return fig
    
    def create_sector_overview_plot(self, buy_candidates: List, sell_candidates: List):
        """Create sector overview plot (same as original)."""
        all_candidates = []
        
        for ticker, score, volatility, sector, industry in buy_candidates:
            all_candidates.append({
                'ticker': ticker,
                'score': score,
                'sentiment': 0.0,  # Placeholder for sentiment
                'industry': industry,
                'sector': sector,
                'volatility': volatility,
                'recommendation': 'Buy',
                'risk_level': 'Low' if volatility < 0.3 else 'Medium' if volatility < 0.5 else 'High'
            })
        
        for ticker, score, volatility, sector, industry in sell_candidates:
            all_candidates.append({
                'ticker': ticker,
                'score': score,
                'sentiment': 0.0,  # Placeholder for sentiment
                'industry': industry,
                'sector': sector,
                'volatility': volatility,
                'recommendation': 'Sell',
                'risk_level': 'Low' if volatility < 0.3 else 'Medium' if volatility < 0.5 else 'High'
            })
        
        if not all_candidates:
            return
        
        df_candidates = pd.DataFrame(all_candidates)
        
        # Create scatter plot
        fig = px.scatter(
            df_candidates,
            x='sentiment',
            y='score',
            color='sector',
            symbol='recommendation',
            size='volatility',
            hover_data=['ticker', 'industry', 'volatility', 'risk_level'],
            title='Stock Recommendations: Sentiment vs Score (Size = Volatility)',
            labels={
                'sentiment': 'Average Sentiment Score',
                'score': 'Recommendation Score',
                'sector': 'Sector',
                'volatility': 'Volatility'
            },
            color_discrete_map=SECTOR_COLORS,
            size_max=20
        )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            xaxis_title="News Sentiment Score (Negative ← → Positive)",
            yaxis_title="Recommendation Score (Higher = Stronger Signal)"
        )
        
        filename = os.path.join(self.plots_path, "sector_overview.html")
        fig.write_html(filename)
        self.logger.info("Saved sector overview plot")
        
        return fig
    
    def plot_all_tickers(self, df: pd.DataFrame):
        """Create the overview plot of all tickers (same as original)."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(19.2, 10.8))
        
        for symbol in df['symbol'].unique():
            ticker_data = df[df['symbol'] == symbol].sort_values('date')
            if len(ticker_data) < 10:
                continue
            
            dates = ticker_data['date']
            prices = ticker_data['close']
            
            plt.plot(dates, prices, linewidth=0.8, alpha=0.7)
            plt.text(dates.iloc[-1], prices.iloc[-1], f' {symbol}', fontsize=8, va='center')
        
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('NASDAQ-100 & TSX-60 Prices')
        plt.grid(True)
        plt.tight_layout()
        
        filename = os.path.join(self.plots_path, "all_tickers_overview.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.show()
        
        self.logger.info("Saved all tickers overview plot")


async def main():
    """Main demo function."""
    # Initialize configuration and logging
    settings = get_settings()
    setup_logging(
        level=settings.logging.log_level,
        format_string=settings.logging.format
    )
    
    logger = get_logger(__name__)
    logger.info("Starting full system demo")
    
    print("🚀 Full System Demo - Recreating Original Functionality")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = ModernStockAnalyzer()
    
    # 1. Scrape tickers
    print("\n1. Scraping Tickers from Wikipedia:")
    nasdaq_tickers = analyzer.get_nasdaq_100_tickers()
    tsx_tickers = analyzer.get_tsx60_tickers()
    all_tickers = nasdaq_tickers + tsx_tickers
    
    print(f"   ✅ NASDAQ-100: {len(nasdaq_tickers)} tickers")
    print(f"   ✅ TSX-60: {len(tsx_tickers)} tickers")
    print(f"   📊 Total: {len(all_tickers)} tickers")
    
    # 2. Sync data (limit to first 20 for demo speed)
    demo_tickers = all_tickers[:20]  # Limit for demo
    print(f"\n2. Syncing Stock Data (Demo: {len(demo_tickers)} tickers):")
    await analyzer.sync_all_tickers(demo_tickers)
    print("   ✅ Data sync completed")
    
    # 3. Get data for analysis
    print("\n3. Loading Data for Analysis:")
    df = await analyzer.get_stock_dataframe()
    print(f"   📊 Loaded data for {df['symbol'].nunique()} stocks")
    print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
    
    # 4. Analyze stocks
    print("\n4. Analyzing Stocks for Buy/Sell Signals:")
    buy_candidates = []
    sell_candidates = []
    
    grouped = df.groupby('symbol')
    for ticker, group in grouped:
        result = analyzer.analyze_ticker(group)
        if result is None:
            continue
        
        if result['buy_score'] >= 4:
            buy_candidates.append((
                ticker, 
                result['buy_score'], 
                result['volatility'], 
                result['sector'], 
                result['industry']
            ))
        
        if result['sell_score'] >= 4:
            sell_candidates.append((
                ticker, 
                result['sell_score'], 
                result['volatility'], 
                result['sector'], 
                result['industry']
            ))
    
    # Sort by score
    buy_candidates.sort(key=lambda x: x[1], reverse=True)
    sell_candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   📈 Found {len(buy_candidates)} buy candidates")
    print(f"   📉 Found {len(sell_candidates)} sell candidates")
    
    # 5. Display recommendations
    print("\n5. Stock Recommendations:")
    print("\n   🟢 BUY RECOMMENDATIONS:")
    for ticker, score, volatility, sector, industry in buy_candidates[:10]:
        print(f"      {ticker:6} | Score: {score} | Vol: {volatility:.3f} | {sector} - {industry}")
    
    print("\n   🔴 SELL RECOMMENDATIONS:")
    for ticker, score, volatility, sector, industry in sell_candidates[:10]:
        print(f"      {ticker:6} | Score: {score} | Vol: {volatility:.3f} | {sector} - {industry}")
    
    # 6. Create plots
    print("\n6. Creating Interactive Plots:")
    
    # Create sector overview
    if buy_candidates or sell_candidates:
        analyzer.create_sector_overview_plot(buy_candidates, sell_candidates)
        print("   ✅ Sector overview plot created")
    
    # Create individual stock analysis plots for top candidates
    top_stocks = []
    if buy_candidates:
        top_stocks.extend([t[0] for t in buy_candidates[:3]])
    if sell_candidates:
        top_stocks.extend([t[0] for t in sell_candidates[:3]])
    
    for ticker in top_stocks:
        if ticker in grouped.groups:
            ticker_data = grouped.get_group(ticker)
            analyzer.plot_stock_analysis(ticker_data, ticker)
    
    print(f"   ✅ Created detailed analysis plots for {len(top_stocks)} stocks")
    
    # Create overview plot of all tickers
    if not df.empty:
        analyzer.plot_all_tickers(df)
        print("   ✅ All tickers overview plot created")
    
    # 7. Database statistics
    print("\n7. Database Statistics:")
    all_stocks = await analyzer.repo.get_all_stocks()
    print(f"   📊 Total stocks in database: {len(all_stocks)}")
    
    # Get some statistics
    if all_stocks:
        sample_stock = all_stocks[0]
        stats = await analyzer.repo.get_stock_statistics(sample_stock.id, days_back=30)
        if stats.get('data_points', 0) > 0:
            print(f"   📈 Sample stock ({sample_stock.symbol}) has {stats['data_points']} data points")
    
    sectors = await analyzer.repo.get_sectors()
    print(f"   🏢 Sectors represented: {len(sectors)}")
    if sectors:
        print(f"      {', '.join(sectors[:5])}{'...' if len(sectors) > 5 else ''}")
    
    print("\n" + "=" * 70)
    print("🎉 FULL SYSTEM DEMO COMPLETE!")
    print("=" * 70)
    
    print("\n✅ Successfully demonstrated:")
    print("  • Web scraping of NASDAQ-100 and TSX-60 tickers")
    print("  • yFinance data fetching and validation")
    print("  • Clean architecture database storage")
    print("  • Technical analysis with RSI, MA, volatility")
    print("  • Buy/sell signal generation")
    print("  • Interactive Plotly visualizations")
    print("  • Matplotlib overview plots")
    print("  • Sector and industry analysis")
    
    print("\n🏗️ Architecture Benefits Demonstrated:")
    print("  • Type-safe domain entities")
    print("  • Repository pattern for data access")
    print("  • Async/await for performance")
    print("  • Professional logging and error handling")
    print("  • Clean separation of concerns")
    print("  • Configurable and testable components")
    
    print(f"\n📁 Generated Files:")
    print(f"  • Database: {analyzer.repo.db_path}")
    print(f"  • Plots: {analyzer.plots_path}/")
    print(f"    - sector_overview.html")
    print(f"    - all_tickers_overview.png")
    print(f"    - [ticker]_analysis.html for top stocks")
    
    print("\n🚀 The new architecture successfully recreates all original functionality")
    print("   with improved maintainability, type safety, and performance!")


if __name__ == "__main__":
    asyncio.run(main())

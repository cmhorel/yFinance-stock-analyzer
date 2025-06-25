"""Application service for stock data management."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal

import yfinance as yf
import pandas as pd

from ...domain.entities.stock import Stock, StockPrice, StockInfo
from ...domain.repositories.stock_repository import IStockRepository
from ...shared.logging import get_logger


class StockDataService:
    """Application service for managing stock data operations."""
    
    def __init__(self, stock_repository: IStockRepository):
        self.stock_repo = stock_repository
        self.logger = get_logger(__name__)
    
    def get_nasdaq_100_tickers(self) -> List[str]:
        """Scrape NASDAQ-100 tickers from Wikipedia."""
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')
            tickers = tables[4]['Ticker'].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
            return tickers
        except Exception as e:
            self.logger.error(f"Error scraping NASDAQ-100 tickers: {e}")
            return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'AMD']
    
    def get_tsx_tickers(self) -> List[str]:
        """Get major TSX tickers."""
        tsx_tickers = [
            'SHOP.TO', 'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
            'CNR.TO', 'CP.TO', 'ENB.TO', 'TRP.TO', 'SU.TO', 'CNQ.TO',
            'WCN.TO', 'CSU.TO', 'ATD.TO', 'L.TO', 'MFC.TO', 'SLF.TO',
            'ABX.TO', 'WEED.TO', 'ACB.TO', 'BB.TO', 'BAM.TO'
        ]
        return tsx_tickers
    
    async def load_all_stocks_on_startup(self) -> Dict[str, Any]:
        """Load all NASDAQ and TSX stocks on application startup."""
        self.logger.info("Loading all stocks on startup...")
        
        # Get tickers
        nasdaq_symbols = self.get_nasdaq_100_tickers()
        tsx_symbols = self.get_tsx_tickers()
        all_symbols = nasdaq_symbols + tsx_symbols
        
        self.logger.info(f"Loading {len(nasdaq_symbols)} NASDAQ and {len(tsx_symbols)} TSX stocks...")
        
        # Load stocks in batches to avoid overwhelming the API
        batch_size = 10
        loaded_count = 0
        failed_count = 0
        
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i + batch_size]
            self.logger.info(f"Loading batch {i//batch_size + 1}/{(len(all_symbols) + batch_size - 1)//batch_size}")
            
            tasks = [self.sync_ticker_data(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                elif result:
                    loaded_count += 1
                else:
                    failed_count += 1
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        self.logger.info(f"Stock loading complete: {loaded_count} loaded, {failed_count} failed")
        
        return {
            'nasdaq_count': len(nasdaq_symbols),
            'tsx_count': len(tsx_symbols),
            'total_symbols': len(all_symbols),
            'loaded_count': loaded_count,
            'failed_count': failed_count
        }
    
    async def sync_ticker_data(self, symbol: str) -> bool:
        """Sync data for a single ticker."""
        try:
            stock = await self.stock_repo.get_stock_by_symbol(symbol)
            if not stock:
                exchange = "TSX" if ".TO" in symbol else "NASDAQ"
                stock = await self.stock_repo.create_stock(Stock(
                    id=None, symbol=symbol, name=None, exchange=exchange
                ))
            
            latest_date = await self.stock_repo.get_latest_stock_price_date(stock.id)
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
                await self.stock_repo.create_stock_prices(prices)
            
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
                    await self.stock_repo.create_or_update_stock_info(stock_info)
            except:
                pass
            
            return True
        except Exception as e:
            self.logger.error(f"Error syncing {symbol}: {e}")
            return False
    
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol."""
        try:
            # First try to get from database (latest price)
            stock = await self.stock_repo.get_stock_by_symbol(symbol)
            if stock:
                prices = await self.stock_repo.get_stock_prices(stock.id, limit=1)
                if prices:
                    return prices[0].close_price
            
            # Fallback to yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return Decimal(str(hist['Close'].iloc[-1]))
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None

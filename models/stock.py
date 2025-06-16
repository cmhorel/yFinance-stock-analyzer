# models/stock.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import pandas as pd

@dataclass
class Stock:
    id: Optional[int]
    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None

@dataclass
class StockPrice:
    stock_id: int
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

@dataclass
class NewsItem:
    stock_id: int
    date: str
    title: str
    summary: Optional[str]
    sentiment_score: float
    confidence_score: Optional[float] = None
    source: Optional[str] = None

@dataclass
class TechnicalIndicators:
    stock_id: int
    date: str
    rsi_14: Optional[float] = None
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    ma_200: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    volume_sma_20: Optional[float] = None

@dataclass
class PortfolioHolding:
    stock_id: int
    symbol: str
    quantity: int
    avg_cost_per_share: float
    total_cost: float
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None

@dataclass
class Transaction:
    stock_id: int
    symbol: str
    transaction_type: str  # 'BUY' or 'SELL'
    quantity: int
    price_per_share: float
    total_amount: float
    brokerage_fee: float
    transaction_date: str
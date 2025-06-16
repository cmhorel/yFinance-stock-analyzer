# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    name: str = 'data/stocks.db'
    timeout: int = 30
    check_same_thread: bool = False

@dataclass
class TradingConfig:
    initial_capital: float = 10000.0
    brokerage_fee: float = 10.0
    max_position_size: float = 0.15  # 15% of portfolio
    min_transaction_interval_days: int = 7
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

@dataclass
class ScrapingConfig:
    threads: int = 10
    batch_size: int = 5
    lookback_months: int = 6
    news_lookback_days: int = 7

@dataclass
class Config:
    database: DatabaseConfig = DatabaseConfig()
    trading: TradingConfig = TradingConfig()
    scraping: ScrapingConfig = ScrapingConfig()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

config = Config()
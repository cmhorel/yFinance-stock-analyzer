"""Repository interfaces for the stock analyzer application."""
from .stock_repository import IStockRepository
from .news_repository import INewsRepository
from .portfolio_repository import IPortfolioRepository
from .analysis_repository import IAnalysisRepository

__all__ = [
    'IStockRepository',
    'INewsRepository', 
    'IPortfolioRepository',
    'IAnalysisRepository'
]

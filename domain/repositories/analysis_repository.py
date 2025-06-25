"""Analysis repository interface."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from decimal import Decimal

from ..entities.analysis import AnalysisResult, TechnicalIndicators, Signal, SignalType, SignalStrength


class IAnalysisRepository(ABC):
    """Interface for analysis data repository operations."""
    
    # Technical Indicators Operations
    @abstractmethod
    async def get_technical_indicators(
        self, 
        stock_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[TechnicalIndicators]:
        """Get technical indicators for a stock within date range."""
        pass
    
    @abstractmethod
    async def get_latest_technical_indicators(self, stock_id: int) -> Optional[TechnicalIndicators]:
        """Get the most recent technical indicators for a stock."""
        pass
    
    @abstractmethod
    async def create_technical_indicators(self, indicators: TechnicalIndicators) -> TechnicalIndicators:
        """Create new technical indicators record."""
        pass
    
    @abstractmethod
    async def bulk_create_technical_indicators(self, indicators_list: List[TechnicalIndicators]) -> List[TechnicalIndicators]:
        """Create multiple technical indicators records in bulk."""
        pass
    
    @abstractmethod
    async def update_technical_indicators(self, indicators: TechnicalIndicators) -> TechnicalIndicators:
        """Update existing technical indicators."""
        pass
    
    @abstractmethod
    async def delete_technical_indicators(self, stock_id: int, date: date) -> bool:
        """Delete technical indicators for a specific stock and date."""
        pass
    
    # Signal Operations
    @abstractmethod
    async def get_signal_by_id(self, signal_id: int) -> Optional[Signal]:
        """Get a signal by its ID."""
        pass
    
    @abstractmethod
    async def get_signals_for_stock(
        self, 
        stock_id: int,
        signal_type: Optional[SignalType] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[Signal]:
        """Get signals for a specific stock with optional filtering."""
        pass
    
    @abstractmethod
    async def get_latest_signal(self, stock_id: int, signal_type: Optional[SignalType] = None) -> Optional[Signal]:
        """Get the most recent signal for a stock."""
        pass
    
    @abstractmethod
    async def create_signal(self, signal: Signal) -> Signal:
        """Create a new signal."""
        pass
    
    @abstractmethod
    async def bulk_create_signals(self, signals: List[Signal]) -> List[Signal]:
        """Create multiple signals in bulk."""
        pass
    
    @abstractmethod
    async def update_signal(self, signal: Signal) -> Signal:
        """Update an existing signal."""
        pass
    
    @abstractmethod
    async def delete_signal(self, signal_id: int) -> bool:
        """Delete a signal."""
        pass
    
    @abstractmethod
    async def get_active_signals(
        self, 
        signal_type: Optional[SignalType] = None,
        min_strength: Optional[SignalStrength] = None,
        min_confidence: Optional[Decimal] = None,
        limit: Optional[int] = None
    ) -> List[Signal]:
        """Get currently active/actionable signals."""
        pass
    
    # Analysis Results Operations
    @abstractmethod
    async def get_analysis_result_by_id(self, analysis_id: int) -> Optional[AnalysisResult]:
        """Get an analysis result by its ID."""
        pass
    
    @abstractmethod
    async def get_analysis_results_for_stock(
        self, 
        stock_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[AnalysisResult]:
        """Get analysis results for a specific stock."""
        pass
    
    @abstractmethod
    async def get_latest_analysis_result(self, stock_id: int) -> Optional[AnalysisResult]:
        """Get the most recent analysis result for a stock."""
        pass
    
    @abstractmethod
    async def create_analysis_result(self, result: AnalysisResult) -> AnalysisResult:
        """Create a new analysis result."""
        pass
    
    @abstractmethod
    async def bulk_create_analysis_results(self, results: List[AnalysisResult]) -> List[AnalysisResult]:
        """Create multiple analysis results in bulk."""
        pass
    
    @abstractmethod
    async def update_analysis_result(self, result: AnalysisResult) -> AnalysisResult:
        """Update an existing analysis result."""
        pass
    
    @abstractmethod
    async def delete_analysis_result(self, analysis_id: int) -> bool:
        """Delete an analysis result."""
        pass
    
    # Analytics and Aggregations
    @abstractmethod
    async def get_signal_statistics(
        self, 
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get signal generation statistics."""
        pass
    
    @abstractmethod
    async def get_signal_performance(
        self, 
        signal_type: SignalType,
        days_back: int = 90
    ) -> Dict[str, Any]:
        """Get performance statistics for signals (accuracy, returns, etc.)."""
        pass
    
    @abstractmethod
    async def get_top_signals(
        self, 
        signal_type: SignalType,
        days_back: int = 7,
        limit: int = 10
    ) -> List[Signal]:
        """Get top signals by score and confidence."""
        pass
    
    @abstractmethod
    async def get_stocks_with_signals(
        self, 
        signal_type: SignalType,
        min_score: Optional[int] = None,
        min_confidence: Optional[Decimal] = None
    ) -> List[Dict[str, Any]]:
        """Get stocks that currently have specific types of signals."""
        pass
    
    @abstractmethod
    async def get_sector_analysis_summary(
        self, 
        sector: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get analysis summary for all stocks in a sector."""
        pass
    
    @abstractmethod
    async def get_industry_analysis_summary(
        self, 
        industry: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get analysis summary for all stocks in an industry."""
        pass
    
    # Technical Analysis Aggregations
    @abstractmethod
    async def get_rsi_distribution(
        self, 
        days_back: int = 30
    ) -> Dict[str, int]:
        """Get distribution of RSI values across all stocks."""
        pass
    
    @abstractmethod
    async def get_volatility_leaders(
        self, 
        days_back: int = 30,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get stocks with highest volatility."""
        pass
    
    @abstractmethod
    async def get_momentum_leaders(
        self, 
        days_back: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get stocks with highest momentum."""
        pass
    
    @abstractmethod
    async def get_oversold_stocks(
        self, 
        rsi_threshold: Decimal = Decimal('30'),
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get stocks that are currently oversold."""
        pass
    
    @abstractmethod
    async def get_overbought_stocks(
        self, 
        rsi_threshold: Decimal = Decimal('70'),
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get stocks that are currently overbought."""
        pass
    
    # Historical Analysis
    @abstractmethod
    async def get_signal_history(
        self, 
        stock_id: int,
        signal_type: SignalType,
        days_back: int = 365
    ) -> List[Signal]:
        """Get historical signals for backtesting and analysis."""
        pass
    
    @abstractmethod
    async def get_analysis_trend(
        self, 
        stock_id: int,
        metric: str,  # e.g., 'rsi', 'volatility', 'sentiment'
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """Get trend data for a specific analysis metric."""
        pass
    
    # Cleanup and Maintenance
    @abstractmethod
    async def cleanup_old_indicators(self, days_old: int = 365) -> int:
        """Remove old technical indicators. Returns count of deleted records."""
        pass
    
    @abstractmethod
    async def cleanup_old_signals(self, days_old: int = 90) -> int:
        """Remove old signals. Returns count of deleted records."""
        pass
    
    @abstractmethod
    async def cleanup_old_analysis_results(self, days_old: int = 180) -> int:
        """Remove old analysis results. Returns count of deleted records."""
        pass
    
    # Batch Processing Support
    @abstractmethod
    async def get_stocks_needing_analysis(
        self, 
        hours_since_last_analysis: int = 24
    ) -> List[int]:
        """Get stock IDs that need fresh analysis."""
        pass
    
    @abstractmethod
    async def mark_analysis_complete(self, stock_id: int, analysis_date: datetime) -> bool:
        """Mark that analysis has been completed for a stock."""
        pass
    
    @abstractmethod
    async def get_analysis_queue_status(self) -> Dict[str, Any]:
        """Get status of analysis processing queue."""
        pass

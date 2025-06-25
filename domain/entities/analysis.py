"""Analysis-related domain entities."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
from enum import Enum


class SignalType(Enum):
    """Enumeration for signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalStrength(Enum):
    """Enumeration for signal strength."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


@dataclass
class TechnicalIndicators:
    """Represents technical analysis indicators for a stock."""
    stock_id: int
    date: datetime
    
    # Moving averages
    ma_20: Optional[Decimal] = None
    ma_50: Optional[Decimal] = None
    ma_200: Optional[Decimal] = None
    
    # RSI
    rsi: Optional[Decimal] = None
    
    # Volatility
    volatility: Optional[Decimal] = None
    
    # Momentum
    momentum_7d: Optional[Decimal] = None
    momentum_30d: Optional[Decimal] = None
    
    # Volume indicators
    volume_ratio: Optional[Decimal] = None
    avg_volume_20d: Optional[int] = None
    
    # Price indicators
    price_range: Optional[Decimal] = None
    daily_return: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate indicator values."""
        if self.rsi is not None and not (0 <= self.rsi <= 100):
            raise ValueError("RSI must be between 0 and 100")
        
        if self.volatility is not None and self.volatility < 0:
            raise ValueError("Volatility cannot be negative")
        
        if self.volume_ratio is not None and self.volume_ratio < 0:
            raise ValueError("Volume ratio cannot be negative")
    
    @property
    def is_oversold(self) -> bool:
        """Check if RSI indicates oversold condition."""
        return self.rsi is not None and self.rsi < 30
    
    @property
    def is_overbought(self) -> bool:
        """Check if RSI indicates overbought condition."""
        return self.rsi is not None and self.rsi > 70
    
    @property
    def is_uptrend(self) -> bool:
        """Check if moving averages indicate uptrend."""
        if self.ma_20 is None or self.ma_50 is None:
            return False
        return self.ma_20 > self.ma_50
    
    @property
    def is_downtrend(self) -> bool:
        """Check if moving averages indicate downtrend."""
        if self.ma_20 is None or self.ma_50 is None:
            return False
        return self.ma_20 < self.ma_50


@dataclass
class Signal:
    """Represents a trading signal."""
    id: Optional[int]
    stock_id: int
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    score: int
    confidence: Decimal  # 0.0 to 1.0
    price: Decimal
    generated_date: datetime
    reasons: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate signal data."""
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.score < 0:
            raise ValueError("Score cannot be negative")
        
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Symbol cannot be empty")
        
        self.symbol = self.symbol.upper().strip()
        
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_buy_signal(self) -> bool:
        """Check if this is a buy signal."""
        return self.signal_type == SignalType.BUY
    
    @property
    def is_sell_signal(self) -> bool:
        """Check if this is a sell signal."""
        return self.signal_type == SignalType.SELL
    
    @property
    def is_hold_signal(self) -> bool:
        """Check if this is a hold signal."""
        return self.signal_type == SignalType.HOLD
    
    @property
    def is_strong(self) -> bool:
        """Check if this is a strong signal."""
        return self.strength == SignalStrength.STRONG
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (strong enough to act on)."""
        return self.strength in [SignalStrength.MODERATE, SignalStrength.STRONG] and self.confidence >= Decimal('0.6')


@dataclass
class AnalysisResult:
    """Represents the complete analysis result for a stock."""
    id: Optional[int]
    stock_id: int
    symbol: str
    analysis_date: datetime
    
    # Technical indicators
    indicators: TechnicalIndicators
    
    # Generated signals
    buy_signal: Optional[Signal] = None
    sell_signal: Optional[Signal] = None
    
    # Sentiment analysis
    avg_sentiment: Optional[Decimal] = None
    sentiment_count: Optional[int] = None
    
    # Industry comparison
    industry_momentum: Optional[Decimal] = None
    relative_momentum: Optional[Decimal] = None
    
    # Risk metrics
    risk_score: Optional[Decimal] = None
    volatility_rank: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize analysis result."""
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Symbol cannot be empty")
        
        self.symbol = self.symbol.upper().strip()
        
        if self.metadata is None:
            self.metadata = {}
        
        if self.avg_sentiment is not None and not (-1 <= self.avg_sentiment <= 1):
            raise ValueError("Average sentiment must be between -1.0 and 1.0")
        
        if self.risk_score is not None and not (0 <= self.risk_score <= 10):
            raise ValueError("Risk score must be between 0 and 10")
    
    @property
    def has_buy_signal(self) -> bool:
        """Check if analysis generated a buy signal."""
        return self.buy_signal is not None and self.buy_signal.is_actionable
    
    @property
    def has_sell_signal(self) -> bool:
        """Check if analysis generated a sell signal."""
        return self.sell_signal is not None and self.sell_signal.is_actionable
    
    @property
    def primary_signal(self) -> Optional[Signal]:
        """Get the primary (strongest) signal."""
        signals = [s for s in [self.buy_signal, self.sell_signal] if s is not None]
        if not signals:
            return None
        
        # Return signal with highest score and confidence
        return max(signals, key=lambda s: (s.score, s.confidence))
    
    @property
    def sentiment_category(self) -> str:
        """Get sentiment category description."""
        if self.avg_sentiment is None:
            return "unknown"
        
        if self.avg_sentiment > Decimal('0.1'):
            return "positive"
        elif self.avg_sentiment < Decimal('-0.1'):
            return "negative"
        else:
            return "neutral"
    
    @property
    def risk_category(self) -> str:
        """Get risk category description."""
        if self.risk_score is None:
            return "unknown"
        
        if self.risk_score <= 3:
            return "low"
        elif self.risk_score <= 6:
            return "moderate"
        else:
            return "high"
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the analysis result."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)

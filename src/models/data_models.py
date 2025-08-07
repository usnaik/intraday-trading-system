"""
Data models for the Intraday Trading System
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum
import pandas as pd


class MarketSegment(Enum):
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    ENERGY = "energy"
    CONSUMER_GOODS = "consumer_goods"
    RETAIL = "retail"
    TELECOMMUNICATIONS = "telecommunications"
    INDUSTRIALS = "industrials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AnalysisType(Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"


@dataclass
class StockData:
    """Basic stock data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    segment: MarketSegment


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    symbol: str
    timestamp: datetime
    
    # Trend indicators
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    adx: Optional[float] = None
    
    # Momentum indicators
    rsi: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    cci: Optional[float] = None
    williams_r: Optional[float] = None
    
    # Volatility indicators
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    atr: Optional[float] = None
    
    # Volume indicators
    obv: Optional[float] = None
    volume_sma: Optional[float] = None
    vwap: Optional[float] = None


@dataclass
class FundamentalMetrics:
    """Fundamental analysis metrics"""
    symbol: str
    timestamp: datetime
    
    # Profitability
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    
    # Valuation
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    
    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    
    # Financial Health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    
    # Market data
    market_cap: Optional[float] = None
    beta: Optional[float] = None


@dataclass
class SentimentData:
    """Sentiment analysis data"""
    symbol: str
    timestamp: datetime
    
    # Sentiment scores (-1 to 1)
    twitter_sentiment: Optional[float] = None
    news_sentiment: Optional[float] = None
    reddit_sentiment: Optional[float] = None
    
    # Volume of mentions
    twitter_mentions: Optional[int] = None
    news_mentions: Optional[int] = None
    reddit_mentions: Optional[int] = None
    
    # Composite sentiment
    composite_sentiment: Optional[float] = None
    sentiment_strength: Optional[float] = None  # Confidence level


@dataclass
class TradingSignal:
    """Trading signal with analysis breakdown"""
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    
    # Individual analysis scores (0-100)
    fundamental_score: Optional[float] = None
    technical_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    
    # Composite score (0-100)
    composite_score: float = 0.0
    confidence_level: float = 0.0
    
    # Risk metrics
    risk_score: float = 0.0
    volatility: Optional[float] = None
    
    # Supporting data
    reasons: List[str] = field(default_factory=list)
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Portfolio:
    """Portfolio tracking"""
    timestamp: datetime
    total_value: float
    available_cash: float
    positions: Dict[str, 'Position'] = field(default_factory=dict)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0


@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    entry_timestamp: datetime
    entry_price: float
    quantity: int
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class MarketCondition:
    """Overall market condition assessment"""
    timestamp: datetime
    market_trend: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    volatility_level: str  # "HIGH", "MEDIUM", "LOW"
    volume_profile: str  # "HIGH", "NORMAL", "LOW"
    sector_rotation: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    symbol: str
    timestamp: datetime
    
    # Risk measures
    var_1d: Optional[float] = None  # Value at Risk (1 day)
    cvar_1d: Optional[float] = None  # Conditional VaR
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    # Correlation
    market_correlation: Optional[float] = None
    sector_correlation: Optional[float] = None
    
    # Liquidity
    avg_daily_volume: Optional[int] = None
    bid_ask_spread: Optional[float] = None


class DatabaseModels:
    """Database table definitions"""
    
    STOCK_DATA_TABLE = """
    CREATE TABLE IF NOT EXISTS stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER NOT NULL,
        segment TEXT NOT NULL,
        UNIQUE(symbol, timestamp)
    )
    """
    
    TECHNICAL_INDICATORS_TABLE = """
    CREATE TABLE IF NOT EXISTS technical_indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        sma_20 REAL, sma_50 REAL, ema_12 REAL, ema_26 REAL,
        macd REAL, macd_signal REAL, adx REAL,
        rsi REAL, stoch_k REAL, stoch_d REAL, cci REAL, williams_r REAL,
        bb_upper REAL, bb_middle REAL, bb_lower REAL, atr REAL,
        obv REAL, volume_sma REAL, vwap REAL,
        UNIQUE(symbol, timestamp)
    )
    """
    
    FUNDAMENTAL_METRICS_TABLE = """
    CREATE TABLE IF NOT EXISTS fundamental_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        roe REAL, roa REAL, profit_margin REAL, operating_margin REAL,
        pe_ratio REAL, pb_ratio REAL, ev_ebitda REAL,
        revenue_growth REAL, earnings_growth REAL,
        debt_to_equity REAL, current_ratio REAL, quick_ratio REAL,
        market_cap REAL, beta REAL,
        UNIQUE(symbol, timestamp)
    )
    """
    
    SENTIMENT_DATA_TABLE = """
    CREATE TABLE IF NOT EXISTS sentiment_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        twitter_sentiment REAL, news_sentiment REAL, reddit_sentiment REAL,
        twitter_mentions INTEGER, news_mentions INTEGER, reddit_mentions INTEGER,
        composite_sentiment REAL, sentiment_strength REAL,
        UNIQUE(symbol, timestamp)
    )
    """
    
    TRADING_SIGNALS_TABLE = """
    CREATE TABLE IF NOT EXISTS trading_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        signal_type TEXT NOT NULL,
        fundamental_score REAL, technical_score REAL, sentiment_score REAL,
        composite_score REAL NOT NULL,
        confidence_level REAL NOT NULL,
        risk_score REAL NOT NULL,
        target_price REAL, stop_loss REAL, take_profit REAL,
        reasons TEXT
    )
    """
    
    PORTFOLIO_TABLE = """
    CREATE TABLE IF NOT EXISTS portfolio_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        total_value REAL NOT NULL,
        available_cash REAL NOT NULL,
        daily_pnl REAL NOT NULL,
        total_pnl REAL NOT NULL
    )
    """
    
    POSITIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        entry_timestamp DATETIME NOT NULL,
        entry_price REAL NOT NULL,
        quantity INTEGER NOT NULL,
        current_price REAL,
        unrealized_pnl REAL,
        stop_loss REAL,
        take_profit REAL,
        is_active BOOLEAN DEFAULT TRUE
    )
    """

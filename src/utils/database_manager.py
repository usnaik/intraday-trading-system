"""
Database Manager for the Intraday Trading System
Handles data persistence and retrieval
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio
from loguru import logger
import os

from ..models.data_models import (
    TradingSignal, StockData, TechnicalIndicators, 
    FundamentalMetrics, SentimentData, Portfolio, Position,
    DatabaseModels
)


class DatabaseManager:
    """Database management for trading system data"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.db_config = self.config.get('database', {})
        
        # Database path
        self.db_path = self.db_config.get('path', 'data/trading_system.db')
        self.db_dir = os.path.dirname(self.db_path)
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        
        # Initialize database
        self.connection = None
        self._initialize_database()
        
        logger.info(f"Database manager initialized with database at {self.db_path}")
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Create all tables
            self._create_tables()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _create_tables(self):
        """Create all required database tables"""
        try:
            cursor = self.connection.cursor()
            
            # Create tables using the SQL definitions from DatabaseModels
            tables = [
                DatabaseModels.STOCK_DATA_TABLE,
                DatabaseModels.TECHNICAL_INDICATORS_TABLE,
                DatabaseModels.FUNDAMENTAL_METRICS_TABLE,
                DatabaseModels.SENTIMENT_DATA_TABLE,
                DatabaseModels.TRADING_SIGNALS_TABLE,
                DatabaseModels.PORTFOLIO_TABLE,
                DatabaseModels.POSITIONS_TABLE
            ]
            
            for table_sql in tables:
                cursor.execute(table_sql)
            
            # Create indexes for better performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_timestamp ON stock_data(symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timestamp ON technical_indicators(symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_fundamental_metrics_symbol_timestamp ON fundamental_metrics(symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_sentiment_data_symbol_timestamp ON sentiment_data(symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_timestamp ON trading_signals(symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_trading_signals_signal_type ON trading_signals(signal_type)",
                "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_positions_is_active ON positions(is_active)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            self.connection.commit()
            logger.debug("Database tables and indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    async def store_stock_data(self, stock_data: StockData):
        """Store stock price data"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO stock_data 
                (symbol, timestamp, open, high, low, close, volume, segment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stock_data.symbol,
                stock_data.timestamp,
                stock_data.open,
                stock_data.high,
                stock_data.low,
                stock_data.close,
                stock_data.volume,
                stock_data.segment.value
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing stock data for {stock_data.symbol}: {str(e)}")
    
    async def store_technical_indicators(self, tech_indicators: TechnicalIndicators):
        """Store technical indicators"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO technical_indicators 
                (symbol, timestamp, sma_20, sma_50, ema_12, ema_26, macd, macd_signal, adx,
                 rsi, stoch_k, stoch_d, cci, williams_r, bb_upper, bb_middle, bb_lower, atr,
                 obv, volume_sma, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tech_indicators.symbol,
                tech_indicators.timestamp,
                tech_indicators.sma_20,
                tech_indicators.sma_50,
                tech_indicators.ema_12,
                tech_indicators.ema_26,
                tech_indicators.macd,
                tech_indicators.macd_signal,
                tech_indicators.adx,
                tech_indicators.rsi,
                tech_indicators.stoch_k,
                tech_indicators.stoch_d,
                tech_indicators.cci,
                tech_indicators.williams_r,
                tech_indicators.bb_upper,
                tech_indicators.bb_middle,
                tech_indicators.bb_lower,
                tech_indicators.atr,
                tech_indicators.obv,
                tech_indicators.volume_sma,
                tech_indicators.vwap
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing technical indicators for {tech_indicators.symbol}: {str(e)}")
    
    async def store_fundamental_metrics(self, fund_metrics: FundamentalMetrics):
        """Store fundamental metrics"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO fundamental_metrics 
                (symbol, timestamp, roe, roa, profit_margin, operating_margin,
                 pe_ratio, pb_ratio, ev_ebitda, revenue_growth, earnings_growth,
                 debt_to_equity, current_ratio, quick_ratio, market_cap, beta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fund_metrics.symbol,
                fund_metrics.timestamp,
                fund_metrics.roe,
                fund_metrics.roa,
                fund_metrics.profit_margin,
                fund_metrics.operating_margin,
                fund_metrics.pe_ratio,
                fund_metrics.pb_ratio,
                fund_metrics.ev_ebitda,
                fund_metrics.revenue_growth,
                fund_metrics.earnings_growth,
                fund_metrics.debt_to_equity,
                fund_metrics.current_ratio,
                fund_metrics.quick_ratio,
                fund_metrics.market_cap,
                fund_metrics.beta
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing fundamental metrics for {fund_metrics.symbol}: {str(e)}")
    
    async def store_sentiment_data(self, sentiment: SentimentData):
        """Store sentiment analysis data"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO sentiment_data 
                (symbol, timestamp, twitter_sentiment, news_sentiment, reddit_sentiment,
                 twitter_mentions, news_mentions, reddit_mentions, 
                 composite_sentiment, sentiment_strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sentiment.symbol,
                sentiment.timestamp,
                sentiment.twitter_sentiment,
                sentiment.news_sentiment,
                sentiment.reddit_sentiment,
                sentiment.twitter_mentions,
                sentiment.news_mentions,
                sentiment.reddit_mentions,
                sentiment.composite_sentiment,
                sentiment.sentiment_strength
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing sentiment data for {sentiment.symbol}: {str(e)}")
    
    async def store_trading_signal(self, signal: TradingSignal):
        """Store trading signal"""
        try:
            cursor = self.connection.cursor()
            
            # Convert reasons list to JSON string
            reasons_json = json.dumps(signal.reasons) if signal.reasons else None
            
            cursor.execute("""
                INSERT INTO trading_signals 
                (symbol, timestamp, signal_type, fundamental_score, technical_score, 
                 sentiment_score, composite_score, confidence_level, risk_score,
                 target_price, stop_loss, take_profit, reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol,
                signal.timestamp,
                signal.signal_type.value,
                signal.fundamental_score,
                signal.technical_score,
                signal.sentiment_score,
                signal.composite_score,
                signal.confidence_level,
                signal.risk_score,
                signal.target_price,
                signal.stop_loss,
                signal.take_profit,
                reasons_json
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing trading signal for {signal.symbol}: {str(e)}")
    
    async def store_portfolio_snapshot(self, portfolio: Portfolio):
        """Store portfolio snapshot"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO portfolio_history 
                (timestamp, total_value, available_cash, daily_pnl, total_pnl)
                VALUES (?, ?, ?, ?, ?)
            """, (
                portfolio.timestamp,
                portfolio.total_value,
                portfolio.available_cash,
                portfolio.daily_pnl,
                portfolio.total_pnl
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing portfolio snapshot: {str(e)}")
    
    async def store_position(self, position: Position, is_active: bool = True):
        """Store or update position"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO positions 
                (symbol, entry_timestamp, entry_price, quantity, current_price,
                 unrealized_pnl, stop_loss, take_profit, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol,
                position.entry_timestamp,
                position.entry_price,
                position.quantity,
                position.current_price,
                position.unrealized_pnl,
                position.stop_loss,
                position.take_profit,
                is_active
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing position for {position.symbol}: {str(e)}")
    
    def get_latest_signals(self, limit: int = 50, signal_type: str = None) -> List[Dict]:
        """Get latest trading signals"""
        try:
            cursor = self.connection.cursor()
            
            query = """
                SELECT * FROM trading_signals 
                WHERE 1=1
            """
            params = []
            
            if signal_type:
                query += " AND signal_type = ?"
                params.append(signal_type)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                
                # Parse reasons JSON
                if row_dict['reasons']:
                    row_dict['reasons'] = json.loads(row_dict['reasons'])
                
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting latest signals: {str(e)}")
            return []
    
    def get_stock_data(self, symbol: str, start_date: datetime = None, 
                      end_date: datetime = None) -> pd.DataFrame:
        """Get stock data for a symbol"""
        try:
            query = """
                SELECT * FROM stock_data 
                WHERE symbol = ?
            """
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str, start_date: datetime = None,
                               end_date: datetime = None) -> pd.DataFrame:
        """Get technical indicators for a symbol"""
        try:
            query = """
                SELECT * FROM technical_indicators 
                WHERE symbol = ?
            """
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting technical indicators for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_fundamental_metrics(self, symbol: str, latest_only: bool = True) -> Dict:
        """Get fundamental metrics for a symbol"""
        try:
            if latest_only:
                query = """
                    SELECT * FROM fundamental_metrics 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """
            else:
                query = """
                    SELECT * FROM fundamental_metrics 
                    WHERE symbol = ? 
                    ORDER BY timestamp
                """
            
            cursor = self.connection.cursor()
            cursor.execute(query, (symbol,))
            
            if latest_only:
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
            else:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting fundamental metrics for {symbol}: {str(e)}")
            return {}
    
    def get_sentiment_data(self, symbol: str, start_date: datetime = None,
                          end_date: datetime = None) -> pd.DataFrame:
        """Get sentiment data for a symbol"""
        try:
            query = """
                SELECT * FROM sentiment_data 
                WHERE symbol = ?
            """
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting sentiment data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM positions 
                WHERE is_active = 1 
                ORDER BY entry_timestamp DESC
            """)
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting active positions: {str(e)}")
            return []
    
    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio history"""
        try:
            query = """
                SELECT * FROM portfolio_history 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            """.format(days)
            
            df = pd.read_sql_query(query, self.connection)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting portfolio history: {str(e)}")
            return pd.DataFrame()
    
    def get_signal_performance(self, days: int = 30) -> Dict:
        """Get signal performance statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Get signal counts by type
            cursor.execute("""
                SELECT signal_type, COUNT(*) as count
                FROM trading_signals 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY signal_type
            """.format(days))
            
            signal_counts = dict(cursor.fetchall())
            
            # Get average scores by signal type
            cursor.execute("""
                SELECT signal_type, 
                       AVG(composite_score) as avg_score,
                       AVG(confidence_level) as avg_confidence,
                       AVG(risk_score) as avg_risk
                FROM trading_signals 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY signal_type
            """.format(days))
            
            signal_stats = {}
            for row in cursor.fetchall():
                signal_stats[row[0]] = {
                    'avg_score': row[1],
                    'avg_confidence': row[2],
                    'avg_risk': row[3]
                }
            
            return {
                'signal_counts': signal_counts,
                'signal_stats': signal_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting signal performance: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to save space"""
        try:
            cursor = self.connection.cursor()
            
            # Tables to clean up
            tables = [
                'stock_data',
                'technical_indicators', 
                'sentiment_data',
                'trading_signals'
            ]
            
            cleanup_date = f"datetime('now', '-{days_to_keep} days')"
            
            for table in tables:
                cursor.execute(f"""
                    DELETE FROM {table} 
                    WHERE timestamp < {cleanup_date}
                """)
                
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old records from {table}")
            
            self.connection.commit()
            
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
            
            logger.info("Database cleanup completed")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    async def close(self):
        """Close database connection"""
        try:
            if self.connection:
                self.connection.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            if hasattr(self, 'connection') and self.connection:
                self.connection.close()
        except:
            pass

"""
Main Trading System Core
Combines fundamental, technical, and sentiment analysis to generate trading recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import yaml
from loguru import logger
import os

from ..models.data_models import (
    TradingSignal, SignalType, MarketSegment, Portfolio, Position, 
    StockData, TechnicalIndicators, FundamentalMetrics, SentimentData
)
from ..data.yahoo_data_fetcher import YahooDataFetcher
from ..technical.technical_analyzer import TechnicalAnalyzer
from ..fundamental.fundamental_analyzer import FundamentalAnalyzer
from ..sentiment.sentiment_analyzer import SentimentAnalyzer
from ..utils.database_manager import DatabaseManager
from ..utils.risk_manager import RiskManager


class IntradayTradingSystem:
    """Main trading system that orchestrates all analyses"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the trading system
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_fetcher = YahooDataFetcher()
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.fundamental_analyzer = FundamentalAnalyzer(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.database_manager = DatabaseManager(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Get market segments and stocks
        self.market_segments = self._parse_market_segments()
        self.all_symbols = self._get_all_symbols()
        
        # Analysis weights
        self.analysis_weights = self.config.get('analysis_weights', {
            'fundamental': 0.30,
            'technical': 0.40,
            'sentiment': 0.30
        })
        
        # Trading parameters
        self.trading_config = self.config.get('trading', {})
        
        # Portfolio tracking
        self.portfolio = Portfolio(
            timestamp=datetime.now(),
            total_value=100000.0,  # Default $100k portfolio
            available_cash=100000.0,
            positions={}
        )
        
        logger.info("Intraday Trading System initialized")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load system configuration"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is not available"""
        return {
            'market_segments': {
                'technology': {'stocks': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']},
                'finance': {'stocks': ['JPM', 'BAC', 'WFC', 'GS', 'MS']},
                'healthcare': {'stocks': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK']},
                'energy': {'stocks': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']},
                'retail': {'stocks': ['AMZN', 'TGT', 'HD', 'LOW', 'SBUX']}
            },
            'analysis_weights': {
                'fundamental': 0.30,
                'technical': 0.40,
                'sentiment': 0.30
            },
            'trading': {
                'max_positions': 10,
                'position_size_percent': 10.0,
                'stop_loss_percent': 2.0,
                'take_profit_percent': 4.0
            }
        }
    
    def _parse_market_segments(self) -> Dict[MarketSegment, List[str]]:
        """Parse market segments from configuration"""
        segments = {}
        
        for segment_name, data in self.config.get('market_segments', {}).items():
            try:
                segment_enum = MarketSegment(segment_name)
                segments[segment_enum] = data.get('stocks', [])
            except ValueError:
                logger.warning(f"Unknown market segment: {segment_name}")
                continue
        
        return segments
    
    def _get_all_symbols(self) -> List[str]:
        """Get all unique symbols from all market segments"""
        all_symbols = []
        for stocks in self.market_segments.values():
            all_symbols.extend(stocks)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_symbols))
    
    async def run_analysis(self, symbols: List[str] = None) -> Dict[str, TradingSignal]:
        """
        Run complete analysis for given symbols
        
        Args:
            symbols: List of symbols to analyze (default: all configured symbols)
            
        Returns:
            Dictionary of trading signals
        """
        if symbols is None:
            symbols = self.all_symbols
        
        logger.info(f"Running analysis for {len(symbols)} symbols")
        
        try:
            # Validate symbols
            valid_symbols = self.data_fetcher.validate_symbols(symbols)
            logger.info(f"Validated {len(valid_symbols)} symbols")
            
            # Fetch market data
            logger.info("Fetching market data...")
            market_data = self.data_fetcher.get_stock_data(valid_symbols, period="5d", interval="1m")
            
            # Run analyses concurrently
            logger.info("Running parallel analyses...")
            
            # Technical analysis
            technical_task = self._run_technical_analysis(market_data, valid_symbols)
            
            # Fundamental analysis  
            fundamental_task = self._run_fundamental_analysis(valid_symbols)
            
            # Sentiment analysis
            sentiment_task = self._run_sentiment_analysis(valid_symbols)
            
            # Wait for all analyses to complete
            technical_results, fundamental_results, sentiment_results = await asyncio.gather(
                technical_task, fundamental_task, sentiment_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(technical_results, Exception):
                logger.error(f"Technical analysis failed: {str(technical_results)}")
                technical_results = {}
            
            if isinstance(fundamental_results, Exception):
                logger.error(f"Fundamental analysis failed: {str(fundamental_results)}")
                fundamental_results = {}
            
            if isinstance(sentiment_results, Exception):
                logger.error(f"Sentiment analysis failed: {str(sentiment_results)}")
                sentiment_results = {}
            
            # Combine analyses and generate final signals
            logger.info("Combining analyses and generating signals...")
            combined_signals = await self._combine_analyses(
                technical_results, fundamental_results, sentiment_results, valid_symbols
            )
            
            # Apply risk management
            logger.info("Applying risk management...")
            filtered_signals = await self._apply_risk_management(combined_signals)
            
            # Store results in database
            await self._store_results(filtered_signals, technical_results, fundamental_results, sentiment_results)
            
            logger.info(f"Analysis completed. Generated {len(filtered_signals)} trading signals")
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error in run_analysis: {str(e)}")
            return {}
    
    async def _run_technical_analysis(self, market_data: Dict, symbols: List[str]) -> Dict[str, TradingSignal]:
        """Run technical analysis for all symbols"""
        try:
            technical_signals = {}
            
            for symbol in symbols:
                if symbol in market_data and not market_data[symbol].empty:
                    # Calculate technical indicators
                    data_with_indicators = self.technical_analyzer.calculate_all_indicators(
                        market_data[symbol]
                    )
                    
                    # Generate signals
                    signals = self.technical_analyzer.generate_signals(data_with_indicators, symbol)
                    
                    if signals:
                        technical_signals[symbol] = signals[0]  # Take the first (most recent) signal
                else:
                    logger.warning(f"No market data available for technical analysis of {symbol}")
            
            return technical_signals
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return {}
    
    async def _run_fundamental_analysis(self, symbols: List[str]) -> Dict[str, FundamentalMetrics]:
        """Run fundamental analysis for all symbols"""
        try:
            fundamental_results = await self.fundamental_analyzer.analyze_fundamentals(symbols)
            return fundamental_results
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {str(e)}")
            return {}
    
    async def _run_sentiment_analysis(self, symbols: List[str]) -> Dict[str, SentimentData]:
        """Run sentiment analysis for all symbols"""
        try:
            sentiment_results = await self.sentiment_analyzer.analyze_sentiment(symbols)
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {}
    
    async def _combine_analyses(
        self, 
        technical_results: Dict, 
        fundamental_results: Dict, 
        sentiment_results: Dict,
        symbols: List[str]
    ) -> Dict[str, TradingSignal]:
        """
        Combine all analyses to generate final trading signals
        """
        try:
            combined_signals = {}
            
            for symbol in symbols:
                try:
                    # Get results for this symbol
                    technical_signal = technical_results.get(symbol)
                    fundamental_metrics = fundamental_results.get(symbol)
                    sentiment_data = sentiment_results.get(symbol)
                    
                    # Calculate individual scores
                    technical_score = technical_signal.technical_score if technical_signal else 50.0
                    
                    fundamental_score = 50.0
                    if fundamental_metrics:
                        segment_name = self._get_symbol_segment(symbol)
                        fundamental_score = self.fundamental_analyzer.generate_fundamental_score(
                            fundamental_metrics, segment_name
                        )
                    
                    sentiment_score = 50.0
                    if sentiment_data:
                        sentiment_score = self.sentiment_analyzer.generate_sentiment_score(sentiment_data)
                    
                    # Calculate weighted composite score
                    composite_score = (
                        technical_score * self.analysis_weights['technical'] +
                        fundamental_score * self.analysis_weights['fundamental'] +
                        sentiment_score * self.analysis_weights['sentiment']
                    )
                    
                    # Determine signal type based on composite score
                    signal_type = self._determine_composite_signal_type(
                        composite_score, technical_signal, fundamental_score, sentiment_score
                    )
                    
                    # Calculate confidence and risk
                    confidence = self._calculate_composite_confidence(
                        technical_signal, fundamental_score, sentiment_data
                    )
                    
                    risk_score = self._calculate_composite_risk(
                        technical_signal, fundamental_metrics, sentiment_data
                    )
                    
                    # Generate combined reasons
                    reasons = self._generate_combined_reasons(
                        technical_signal, fundamental_metrics, sentiment_data, 
                        signal_type, fundamental_score
                    )
                    
                    # Calculate price targets (use technical analysis targets if available)
                    target_price = technical_signal.target_price if technical_signal else None
                    stop_loss = technical_signal.stop_loss if technical_signal else None
                    take_profit = technical_signal.take_profit if technical_signal else None
                    
                    # Create combined signal
                    combined_signal = TradingSignal(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        signal_type=signal_type,
                        fundamental_score=fundamental_score,
                        technical_score=technical_score,
                        sentiment_score=sentiment_score,
                        composite_score=composite_score,
                        confidence_level=confidence,
                        risk_score=risk_score,
                        reasons=reasons,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    combined_signals[symbol] = combined_signal
                    
                    logger.debug(f"Combined signal for {symbol}: {signal_type.value} "
                               f"(Score: {composite_score:.2f}, Confidence: {confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error combining analyses for {symbol}: {str(e)}")
                    continue
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error in _combine_analyses: {str(e)}")
            return {}
    
    def _get_symbol_segment(self, symbol: str) -> str:
        """Get the market segment for a symbol"""
        for segment, stocks in self.market_segments.items():
            if symbol in stocks:
                return segment.value
        return 'default'
    
    def _determine_composite_signal_type(
        self, 
        composite_score: float, 
        technical_signal: TradingSignal,
        fundamental_score: float,
        sentiment_score: float
    ) -> SignalType:
        """Determine signal type based on composite analysis"""
        try:
            # Strong composite signals
            if composite_score >= 75:
                return SignalType.BUY
            elif composite_score <= 25:
                return SignalType.SELL
            
            # Check for agreement between analyses
            buy_votes = 0
            sell_votes = 0
            
            # Technical vote
            if technical_signal:
                if technical_signal.signal_type == SignalType.BUY:
                    buy_votes += 1
                elif technical_signal.signal_type == SignalType.SELL:
                    sell_votes += 1
            
            # Fundamental vote
            if fundamental_score >= 65:
                buy_votes += 1
            elif fundamental_score <= 35:
                sell_votes += 1
            
            # Sentiment vote
            if sentiment_score >= 60:
                buy_votes += 1
            elif sentiment_score <= 40:
                sell_votes += 1
            
            # Determine signal based on votes
            if buy_votes >= 2 and buy_votes > sell_votes:
                return SignalType.BUY
            elif sell_votes >= 2 and sell_votes > buy_votes:
                return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            logger.error(f"Error determining composite signal type: {str(e)}")
            return SignalType.HOLD
    
    def _calculate_composite_confidence(
        self,
        technical_signal: TradingSignal,
        fundamental_score: float,
        sentiment_data: SentimentData
    ) -> float:
        """Calculate confidence level for composite signal"""
        try:
            confidence_scores = []
            
            # Technical confidence
            if technical_signal:
                confidence_scores.append(technical_signal.confidence_level)
            
            # Fundamental confidence (based on how extreme the score is)
            fundamental_confidence = 50 + abs(fundamental_score - 50)
            confidence_scores.append(fundamental_confidence)
            
            # Sentiment confidence
            if sentiment_data and sentiment_data.sentiment_strength:
                sentiment_confidence = sentiment_data.sentiment_strength * 100
                confidence_scores.append(sentiment_confidence)
            
            # Return weighted average
            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
            else:
                return 50.0
                
        except Exception as e:
            logger.error(f"Error calculating composite confidence: {str(e)}")
            return 50.0
    
    def _calculate_composite_risk(
        self,
        technical_signal: TradingSignal,
        fundamental_metrics: FundamentalMetrics,
        sentiment_data: SentimentData
    ) -> float:
        """Calculate composite risk score"""
        try:
            risk_scores = []
            
            # Technical risk
            if technical_signal:
                risk_scores.append(technical_signal.risk_score)
            
            # Fundamental risk
            if fundamental_metrics:
                fundamental_risks = self.fundamental_analyzer.get_risk_assessment(fundamental_metrics)
                avg_fundamental_risk = sum(fundamental_risks.values()) / len(fundamental_risks)
                risk_scores.append(avg_fundamental_risk)
            
            # Sentiment risk (low confidence = high risk)
            if sentiment_data and sentiment_data.sentiment_strength:
                sentiment_risk = (1.0 - sentiment_data.sentiment_strength) * 100
                risk_scores.append(sentiment_risk)
            
            # Return weighted average
            if risk_scores:
                return sum(risk_scores) / len(risk_scores)
            else:
                return 50.0
                
        except Exception as e:
            logger.error(f"Error calculating composite risk: {str(e)}")
            return 50.0
    
    def _generate_combined_reasons(
        self,
        technical_signal: TradingSignal,
        fundamental_metrics: FundamentalMetrics,
        sentiment_data: SentimentData,
        signal_type: SignalType,
        fundamental_score: float
    ) -> List[str]:
        """Generate combined reasons from all analyses"""
        try:
            reasons = []
            
            # Technical reasons
            if technical_signal and technical_signal.reasons:
                reasons.extend([f"Technical: {r}" for r in technical_signal.reasons[:2]])
            
            # Fundamental reasons
            if fundamental_metrics:
                segment = self._get_symbol_segment(fundamental_metrics.symbol)
                fund_reasons = self.fundamental_analyzer.get_fundamental_reasons(
                    fundamental_metrics, fundamental_score, segment
                )
                reasons.extend([f"Fundamental: {r}" for r in fund_reasons[:2]])
            
            # Sentiment reasons
            if sentiment_data:
                sent_reasons = self.sentiment_analyzer.get_sentiment_reasons(sentiment_data)
                reasons.extend([f"Sentiment: {r}" for r in sent_reasons[:2]])
            
            # Add overall assessment
            if signal_type == SignalType.BUY:
                reasons.insert(0, "Strong buy recommendation from combined analysis")
            elif signal_type == SignalType.SELL:
                reasons.insert(0, "Strong sell recommendation from combined analysis")
            else:
                reasons.insert(0, "Mixed signals suggest holding position")
            
            return reasons[:6]  # Limit to 6 reasons
            
        except Exception as e:
            logger.error(f"Error generating combined reasons: {str(e)}")
            return ["Combined analysis completed"]
    
    async def _apply_risk_management(self, signals: Dict[str, TradingSignal]) -> Dict[str, TradingSignal]:
        """Apply risk management filters to signals"""
        try:
            filtered_signals = {}
            
            # Sort signals by composite score
            sorted_signals = sorted(
                signals.items(), 
                key=lambda x: x[1].composite_score, 
                reverse=True
            )
            
            buy_signals = []
            sell_signals = []
            
            for symbol, signal in sorted_signals:
                # Apply risk filters
                if not self.risk_manager.is_signal_acceptable(signal, self.portfolio):
                    logger.debug(f"Signal for {symbol} filtered out by risk management")
                    continue
                
                # Separate buy and sell signals
                if signal.signal_type == SignalType.BUY:
                    buy_signals.append((symbol, signal))
                elif signal.signal_type == SignalType.SELL:
                    sell_signals.append((symbol, signal))
                else:
                    filtered_signals[symbol] = signal  # Keep HOLD signals
            
            # Limit number of positions
            max_positions = self.trading_config.get('max_positions', 10)
            
            # Add top buy signals (up to limit)
            for symbol, signal in buy_signals[:max_positions]:
                filtered_signals[symbol] = signal
            
            # Add top sell signals (up to limit)
            for symbol, signal in sell_signals[:max_positions]:
                filtered_signals[symbol] = signal
            
            logger.info(f"Risk management applied. {len(filtered_signals)} signals remaining")
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error in risk management: {str(e)}")
            return signals
    
    async def _store_results(
        self,
        signals: Dict[str, TradingSignal],
        technical_results: Dict,
        fundamental_results: Dict,
        sentiment_results: Dict
    ):
        """Store analysis results in database"""
        try:
            # Store trading signals
            for signal in signals.values():
                await self.database_manager.store_trading_signal(signal)
            
            # Store technical indicators
            for symbol, data in technical_results.items():
                if hasattr(data, 'technical_score'):  # It's a signal with technical data
                    # You would store the underlying technical indicators here
                    pass
            
            # Store fundamental metrics
            for symbol, metrics in fundamental_results.items():
                await self.database_manager.store_fundamental_metrics(metrics)
            
            # Store sentiment data
            for symbol, sentiment in sentiment_results.items():
                await self.database_manager.store_sentiment_data(sentiment)
            
            logger.debug("Analysis results stored in database")
            
        except Exception as e:
            logger.error(f"Error storing results: {str(e)}")
    
    def get_top_recommendations(self, limit: int = 10, signal_type: SignalType = None) -> List[TradingSignal]:
        """
        Get top trading recommendations
        
        Args:
            limit: Maximum number of recommendations
            signal_type: Filter by signal type (optional)
            
        Returns:
            List of top trading signals
        """
        try:
            # This would typically query the database for recent signals
            # For now, we'll return an empty list as a placeholder
            return []
            
        except Exception as e:
            logger.error(f"Error getting top recommendations: {str(e)}")
            return []
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        try:
            return {
                'timestamp': self.portfolio.timestamp.isoformat(),
                'total_value': self.portfolio.total_value,
                'available_cash': self.portfolio.available_cash,
                'positions': len(self.portfolio.positions),
                'daily_pnl': self.portfolio.daily_pnl,
                'total_pnl': self.portfolio.total_pnl
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {str(e)}")
            return {}
    
    def get_market_overview(self) -> Dict:
        """Get market overview with sector performance"""
        try:
            sector_performance = self.data_fetcher.get_sector_performance()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'sector_performance': sector_performance,
                'total_symbols': len(self.all_symbols),
                'market_segments': len(self.market_segments)
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {str(e)}")
            return {}
    
    async def start_realtime_monitoring(self):
        """Start real-time monitoring and analysis"""
        logger.info("Starting real-time monitoring...")
        
        try:
            while True:
                # Run analysis
                signals = await self.run_analysis()
                
                # Process any new signals
                for symbol, signal in signals.items():
                    if signal.signal_type != SignalType.HOLD:
                        logger.info(f"New {signal.signal_type.value} signal for {symbol}: "
                                  f"Score {signal.composite_score:.2f}")
                
                # Wait before next analysis
                analysis_interval = self.config.get('scheduler', {}).get('analysis_interval', 300)
                logger.debug(f"Waiting {analysis_interval} seconds before next analysis")
                await asyncio.sleep(analysis_interval)
                
        except KeyboardInterrupt:
            logger.info("Real-time monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in real-time monitoring: {str(e)}")
    
    async def shutdown(self):
        """Gracefully shutdown the trading system"""
        logger.info("Shutting down trading system...")
        
        try:
            # Close database connections
            await self.database_manager.close()
            
            # Any other cleanup tasks
            logger.info("Trading system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


# Convenience function for creating and running the system
async def run_trading_system(config_path: str = None):
    """
    Convenience function to create and run the trading system
    
    Args:
        config_path: Path to configuration file
    """
    system = IntradayTradingSystem(config_path)
    
    try:
        # Run one-time analysis
        signals = await system.run_analysis()
        
        # Print results
        print(f"\n=== Trading Analysis Results ===")
        print(f"Generated {len(signals)} trading signals")
        
        for symbol, signal in signals.items():
            print(f"\n{symbol}: {signal.signal_type.value}")
            print(f"  Score: {signal.composite_score:.2f}")
            print(f"  Confidence: {signal.confidence_level:.2f}")
            print(f"  Risk: {signal.risk_score:.2f}")
            if signal.reasons:
                print(f"  Reasons: {', '.join(signal.reasons[:3])}")
        
        return system
        
    except Exception as e:
        logger.error(f"Error running trading system: {str(e)}")
        return None
    finally:
        await system.shutdown()

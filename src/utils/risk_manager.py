"""
Risk Management Module for the Intraday Trading System
Implements various risk controls and position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from ..models.data_models import TradingSignal, Portfolio, Position, SignalType


class RiskManager:
    """Risk management engine for trading system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.risk_config = self.config.get('risk_management', {})
        self.trading_config = self.config.get('trading', {})
        
        # Risk parameters
        self.max_daily_loss_percent = self.risk_config.get('max_daily_loss_percent', 5.0)
        self.max_position_correlation = self.risk_config.get('max_position_correlation', 0.7)
        self.volatility_threshold = self.risk_config.get('volatility_threshold', 0.3)
        self.liquidity_min_volume = self.risk_config.get('liquidity_min_volume', 1000000)
        
        # Position sizing
        self.max_positions = self.trading_config.get('max_positions', 10)
        self.position_size_percent = self.trading_config.get('position_size_percent', 10.0)
        self.risk_per_trade_percent = self.trading_config.get('risk_per_trade_percent', 1.0)
        
        # Signal filtering thresholds
        self.min_composite_score = 60.0  # Minimum score for buy signals
        self.min_confidence_level = 50.0  # Minimum confidence required
        self.max_risk_score = 80.0       # Maximum acceptable risk
        
        logger.info("Risk manager initialized")
    
    def is_signal_acceptable(self, signal: TradingSignal, portfolio: Portfolio) -> bool:
        """
        Check if a trading signal meets risk management criteria
        
        Args:
            signal: Trading signal to evaluate
            portfolio: Current portfolio state
            
        Returns:
            True if signal is acceptable, False otherwise
        """
        try:
            logger.debug(f"Evaluating risk for signal {signal.symbol} ({signal.signal_type.value})")
            
            # Check if signal meets minimum quality thresholds
            if not self._meets_quality_thresholds(signal):
                logger.debug(f"Signal {signal.symbol} rejected - quality thresholds not met")
                return False
            
            # Check portfolio-level risks
            if not self._check_portfolio_risk(signal, portfolio):
                logger.debug(f"Signal {signal.symbol} rejected - portfolio risk limits")
                return False
            
            # Check position-level risks
            if not self._check_position_risk(signal, portfolio):
                logger.debug(f"Signal {signal.symbol} rejected - position risk limits")
                return False
            
            # Check correlation limits
            if not self._check_correlation_risk(signal, portfolio):
                logger.debug(f"Signal {signal.symbol} rejected - correlation limits")
                return False
            
            logger.debug(f"Signal {signal.symbol} accepted by risk management")
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating signal risk for {signal.symbol}: {str(e)}")
            return False
    
    def _meets_quality_thresholds(self, signal: TradingSignal) -> bool:
        """Check if signal meets minimum quality thresholds"""
        try:
            # For BUY signals, require higher standards
            if signal.signal_type == SignalType.BUY:
                return (signal.composite_score >= self.min_composite_score and
                        signal.confidence_level >= self.min_confidence_level and
                        signal.risk_score <= self.max_risk_score)
            
            # For SELL signals, we're less strict (risk management for existing positions)
            elif signal.signal_type == SignalType.SELL:
                return (signal.composite_score <= (100 - self.min_composite_score) and
                        signal.confidence_level >= (self.min_confidence_level - 10))
            
            # HOLD signals are generally acceptable
            else:
                return True
                
        except Exception as e:
            logger.error(f"Error checking quality thresholds: {str(e)}")
            return False
    
    def _check_portfolio_risk(self, signal: TradingSignal, portfolio: Portfolio) -> bool:
        """Check portfolio-level risk limits"""
        try:
            # Check daily loss limit
            if portfolio.daily_pnl < -(portfolio.total_value * self.max_daily_loss_percent / 100):
                logger.warning("Daily loss limit reached")
                return False
            
            # Check maximum number of positions
            active_positions = len([p for p in portfolio.positions.values() if p.quantity != 0])
            
            if signal.signal_type == SignalType.BUY and active_positions >= self.max_positions:
                logger.debug("Maximum position limit reached")
                return False
            
            # Check available cash for new positions
            if signal.signal_type == SignalType.BUY:
                position_size = self._calculate_position_size(signal, portfolio)
                if position_size > portfolio.available_cash:
                    logger.debug("Insufficient cash for position")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {str(e)}")
            return False
    
    def _check_position_risk(self, signal: TradingSignal, portfolio: Portfolio) -> bool:
        """Check position-level risk limits"""
        try:
            # Check if we already have a position in this symbol
            existing_position = portfolio.positions.get(signal.symbol)
            
            # For buy signals, check position sizing
            if signal.signal_type == SignalType.BUY:
                # Don't add to existing long positions (for now)
                if existing_position and existing_position.quantity > 0:
                    logger.debug(f"Already have long position in {signal.symbol}")
                    return False
                
                # Check individual position size limit
                position_size = self._calculate_position_size(signal, portfolio)
                max_position_size = portfolio.total_value * self.position_size_percent / 100
                
                if position_size > max_position_size:
                    logger.debug(f"Position size {position_size} exceeds limit {max_position_size}")
                    return False
            
            # For sell signals, we need an existing position
            elif signal.signal_type == SignalType.SELL:
                if not existing_position or existing_position.quantity <= 0:
                    logger.debug(f"No position to sell in {signal.symbol}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position risk: {str(e)}")
            return False
    
    def _check_correlation_risk(self, signal: TradingSignal, portfolio: Portfolio) -> bool:
        """Check correlation limits between positions"""
        try:
            # This is a simplified implementation
            # In a real system, you would calculate actual correlations between assets
            
            # For now, just check sector concentration
            symbol_sector = self._get_symbol_sector(signal.symbol)
            
            same_sector_positions = 0
            total_positions = 0
            
            for symbol, position in portfolio.positions.items():
                if position.quantity > 0:
                    total_positions += 1
                    if self._get_symbol_sector(symbol) == symbol_sector:
                        same_sector_positions += 1
            
            # Don't allow more than 50% of positions in same sector
            if total_positions > 0:
                sector_concentration = same_sector_positions / total_positions
                if sector_concentration > 0.5 and signal.signal_type == SignalType.BUY:
                    logger.debug(f"Sector concentration limit reached for {symbol_sector}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {str(e)}")
            return True  # Default to allowing the trade
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified mapping)"""
        # This is a simplified mapping - in practice, you'd use a proper database
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMZN']
        finance_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
        healthcare_stocks = ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK']
        
        if symbol in tech_stocks:
            return 'technology'
        elif symbol in finance_stocks:
            return 'finance'
        elif symbol in healthcare_stocks:
            return 'healthcare'
        else:
            return 'other'
    
    def _calculate_position_size(self, signal: TradingSignal, portfolio: Portfolio) -> float:
        """Calculate appropriate position size for a signal"""
        try:
            # Base position size as percentage of portfolio
            base_size = portfolio.total_value * self.position_size_percent / 100
            
            # Adjust based on signal strength
            signal_multiplier = signal.composite_score / 100.0
            
            # Adjust based on confidence
            confidence_multiplier = signal.confidence_level / 100.0
            
            # Adjust based on risk (inverse relationship)
            risk_multiplier = max(0.5, (100 - signal.risk_score) / 100.0)
            
            # Calculate final position size
            position_size = base_size * signal_multiplier * confidence_multiplier * risk_multiplier
            
            # Apply risk per trade limit
            max_risk_size = portfolio.total_value * self.risk_per_trade_percent / 100
            
            # Use stop loss to determine actual position size based on risk
            if signal.stop_loss and signal.target_price:
                price_estimate = (signal.target_price + (signal.stop_loss or 0)) / 2
                if price_estimate > 0:
                    risk_per_share = abs(price_estimate - (signal.stop_loss or price_estimate))
                    if risk_per_share > 0:
                        max_shares_by_risk = max_risk_size / risk_per_share
                        risk_based_size = max_shares_by_risk * price_estimate
                        position_size = min(position_size, risk_based_size)
            
            # Ensure we don't exceed available cash
            position_size = min(position_size, portfolio.available_cash * 0.95)  # Leave some buffer
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def calculate_stop_loss(self, signal: TradingSignal, entry_price: float) -> float:
        """Calculate stop loss price"""
        try:
            if signal.stop_loss:
                return signal.stop_loss
            
            # Default stop loss based on ATR or percentage
            if signal.signal_type == SignalType.BUY:
                stop_loss_pct = self.trading_config.get('stop_loss_percent', 2.0)
                return entry_price * (1 - stop_loss_pct / 100)
            elif signal.signal_type == SignalType.SELL:
                stop_loss_pct = self.trading_config.get('stop_loss_percent', 2.0)
                return entry_price * (1 + stop_loss_pct / 100)
            
            return entry_price
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return entry_price
    
    def calculate_take_profit(self, signal: TradingSignal, entry_price: float) -> float:
        """Calculate take profit price"""
        try:
            if signal.take_profit:
                return signal.take_profit
            
            # Default take profit based on risk/reward ratio
            if signal.signal_type == SignalType.BUY:
                take_profit_pct = self.trading_config.get('take_profit_percent', 4.0)
                return entry_price * (1 + take_profit_pct / 100)
            elif signal.signal_type == SignalType.SELL:
                take_profit_pct = self.trading_config.get('take_profit_percent', 4.0)
                return entry_price * (1 - take_profit_pct / 100)
            
            return entry_price
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return entry_price
    
    def validate_order(self, symbol: str, quantity: int, price: float, 
                      signal_type: SignalType, portfolio: Portfolio) -> Dict:
        """
        Validate an order before execution
        
        Returns:
            Dictionary with validation result and details
        """
        try:
            validation_result = {
                'valid': True,
                'reasons': [],
                'warnings': [],
                'adjusted_quantity': quantity,
                'adjusted_price': price
            }
            
            # Check basic parameters
            if quantity <= 0:
                validation_result['valid'] = False
                validation_result['reasons'].append("Invalid quantity")
                return validation_result
            
            if price <= 0:
                validation_result['valid'] = False
                validation_result['reasons'].append("Invalid price")
                return validation_result
            
            # Check order value
            order_value = quantity * price
            
            # For buy orders
            if signal_type == SignalType.BUY:
                if order_value > portfolio.available_cash:
                    validation_result['valid'] = False
                    validation_result['reasons'].append("Insufficient funds")
                
                # Check position size limits
                max_position_value = portfolio.total_value * self.position_size_percent / 100
                if order_value > max_position_value:
                    # Adjust quantity to fit within limits
                    max_quantity = int(max_position_value / price)
                    validation_result['adjusted_quantity'] = max_quantity
                    validation_result['warnings'].append(f"Quantity reduced to {max_quantity} due to position size limits")
            
            # For sell orders
            elif signal_type == SignalType.SELL:
                existing_position = portfolio.positions.get(symbol)
                if not existing_position or existing_position.quantity < quantity:
                    available_quantity = existing_position.quantity if existing_position else 0
                    validation_result['valid'] = False
                    validation_result['reasons'].append(f"Insufficient position (available: {available_quantity})")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating order for {symbol}: {str(e)}")
            return {'valid': False, 'reasons': ['Validation error'], 'warnings': []}
    
    def get_portfolio_risk_metrics(self, portfolio: Portfolio) -> Dict:
        """Calculate portfolio risk metrics"""
        try:
            metrics = {
                'total_exposure': 0.0,
                'cash_percentage': 0.0,
                'position_count': 0,
                'largest_position_pct': 0.0,
                'daily_pnl_pct': 0.0,
                'sector_concentration': {},
                'risk_score': 50.0  # 0-100, higher = more risky
            }
            
            if portfolio.total_value <= 0:
                return metrics
            
            # Calculate basic metrics
            metrics['cash_percentage'] = (portfolio.available_cash / portfolio.total_value) * 100
            metrics['daily_pnl_pct'] = (portfolio.daily_pnl / portfolio.total_value) * 100
            
            # Analyze positions
            position_values = []
            sector_values = {}
            
            for symbol, position in portfolio.positions.items():
                if position.quantity > 0 and position.current_price:
                    position_value = position.quantity * position.current_price
                    position_values.append(position_value)
                    
                    # Sector analysis
                    sector = self._get_symbol_sector(symbol)
                    sector_values[sector] = sector_values.get(sector, 0) + position_value
            
            if position_values:
                metrics['position_count'] = len(position_values)
                metrics['total_exposure'] = sum(position_values)
                metrics['largest_position_pct'] = (max(position_values) / portfolio.total_value) * 100
                
                # Sector concentration
                for sector, value in sector_values.items():
                    metrics['sector_concentration'][sector] = (value / portfolio.total_value) * 100
            
            # Calculate risk score
            risk_factors = []
            
            # Cash level (too high or too low is risky)
            if metrics['cash_percentage'] < 5:
                risk_factors.append(20)  # Too little cash
            elif metrics['cash_percentage'] > 50:
                risk_factors.append(10)  # Too much cash (opportunity cost)
            
            # Position concentration
            if metrics['largest_position_pct'] > 20:
                risk_factors.append(25)
            elif metrics['largest_position_pct'] > 15:
                risk_factors.append(10)
            
            # Sector concentration
            max_sector_pct = max(metrics['sector_concentration'].values()) if metrics['sector_concentration'] else 0
            if max_sector_pct > 60:
                risk_factors.append(20)
            elif max_sector_pct > 40:
                risk_factors.append(10)
            
            # Daily P&L volatility
            if abs(metrics['daily_pnl_pct']) > 3:
                risk_factors.append(15)
            
            metrics['risk_score'] = min(100, sum(risk_factors))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {str(e)}")
            return {'risk_score': 100.0}  # High risk on error
    
    def suggest_position_adjustments(self, portfolio: Portfolio) -> List[Dict]:
        """Suggest position adjustments based on risk analysis"""
        try:
            suggestions = []
            risk_metrics = self.get_portfolio_risk_metrics(portfolio)
            
            # High concentration risk
            if risk_metrics['largest_position_pct'] > 20:
                suggestions.append({
                    'type': 'reduce_concentration',
                    'message': f"Consider reducing largest position (currently {risk_metrics['largest_position_pct']:.1f}% of portfolio)",
                    'priority': 'high'
                })
            
            # Sector concentration risk
            for sector, pct in risk_metrics['sector_concentration'].items():
                if pct > 50:
                    suggestions.append({
                        'type': 'diversify_sector',
                        'message': f"High {sector} concentration ({pct:.1f}%). Consider diversification.",
                        'priority': 'medium'
                    })
            
            # Cash management
            if risk_metrics['cash_percentage'] < 5:
                suggestions.append({
                    'type': 'increase_cash',
                    'message': f"Low cash reserves ({risk_metrics['cash_percentage']:.1f}%). Consider taking some profits.",
                    'priority': 'medium'
                })
            elif risk_metrics['cash_percentage'] > 50:
                suggestions.append({
                    'type': 'deploy_cash',
                    'message': f"High cash reserves ({risk_metrics['cash_percentage']:.1f}%). Consider deploying capital.",
                    'priority': 'low'
                })
            
            # Daily P&L volatility
            if abs(risk_metrics['daily_pnl_pct']) > 5:
                suggestions.append({
                    'type': 'reduce_volatility',
                    'message': f"High daily volatility ({risk_metrics['daily_pnl_pct']:.1f}%). Review position sizing.",
                    'priority': 'high'
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating position adjustment suggestions: {str(e)}")
            return []

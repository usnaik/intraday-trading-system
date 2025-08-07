"""
Technical Analysis Module for the Intraday Trading System
Implements various technical indicators and generates technical scores
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using pandas-based implementations.")

import pandas_ta as ta
from loguru import logger

from ..models.data_models import TechnicalIndicators, SignalType, TradingSignal


class TechnicalAnalyzer:
    """Technical analysis engine for generating trading signals"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.indicators_config = self.config.get('technical_indicators', {})
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given dataset
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            logger.info(f"Calculating technical indicators for {len(df)} data points")
            
            # Make a copy to avoid modifying original data
            data = df.copy()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {required_columns}")
            
            # Calculate trend indicators
            data = self._calculate_trend_indicators(data)
            
            # Calculate momentum indicators
            data = self._calculate_momentum_indicators(data)
            
            # Calculate volatility indicators
            data = self._calculate_volatility_indicators(data)
            
            # Calculate volume indicators
            data = self._calculate_volume_indicators(data)
            
            logger.debug(f"Successfully calculated indicators. DataFrame shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators"""
        try:
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # ADX (Average Directional Index)
            if TALIB_AVAILABLE:
                df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
                df['DI_Plus'] = talib.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
                df['DI_Minus'] = talib.MINUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
            else:
                # ADX calculation without TA-Lib
                df['ADX'], df['DI_Plus'], df['DI_Minus'] = self._calculate_adx_manual(df)
            
            # Parabolic SAR
            if TALIB_AVAILABLE:
                df['SAR'] = talib.SAR(df['High'].values, df['Low'].values, acceleration=0.02, maximum=0.2)
            
            logger.debug("Calculated trend indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
            return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        try:
            # RSI (Relative Strength Index)
            if TALIB_AVAILABLE:
                df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
            else:
                df['RSI'] = self._calculate_rsi_manual(df['Close'], period=14)
            
            # Stochastic Oscillator
            if TALIB_AVAILABLE:
                df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
                    df['High'].values, df['Low'].values, df['Close'].values,
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
            else:
                df['STOCH_K'], df['STOCH_D'] = self._calculate_stochastic_manual(df)
            
            # CCI (Commodity Channel Index)
            if TALIB_AVAILABLE:
                df['CCI'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=20)
            else:
                df['CCI'] = self._calculate_cci_manual(df)
            
            # Williams %R
            if TALIB_AVAILABLE:
                df['Williams_R'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
            else:
                df['Williams_R'] = self._calculate_williams_r_manual(df)
            
            # ROC (Rate of Change)
            df['ROC'] = df['Close'].pct_change(periods=12) * 100
            
            logger.debug("Calculated momentum indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        try:
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
            bb_std_dev = df['Close'].rolling(window=bb_period).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # ATR (Average True Range)
            if TALIB_AVAILABLE:
                df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
            else:
                df['ATR'] = self._calculate_atr_manual(df)
            
            # Historical Volatility
            df['HV_20'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
            
            logger.debug("Calculated volatility indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        try:
            # On-Balance Volume (OBV)
            if TALIB_AVAILABLE:
                df['OBV'] = talib.OBV(df['Close'].values, df['Volume'].values)
            else:
                df['OBV'] = self._calculate_obv_manual(df)
            
            # Volume SMA
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # VWAP (Volume Weighted Average Price)
            df['VWAP'] = self._calculate_vwap(df)
            
            # Money Flow Index (MFI)
            if TALIB_AVAILABLE:
                df['MFI'] = talib.MFI(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values, timeperiod=14)
            else:
                df['MFI'] = self._calculate_mfi_manual(df)
            
            logger.debug("Calculated volume indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return df
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """
        Generate trading signals based on technical indicators
        
        Args:
            df: DataFrame with calculated technical indicators
            symbol: Stock symbol
            
        Returns:
            List of trading signals
        """
        try:
            logger.info(f"Generating technical signals for {symbol}")
            
            signals = []
            
            # Get the latest data point
            if len(df) < 50:  # Need sufficient data for analysis
                logger.warning(f"Insufficient data for {symbol} - need at least 50 data points")
                return signals
            
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            # Calculate individual signal scores
            trend_score = self._calculate_trend_score(df)
            momentum_score = self._calculate_momentum_score(df)
            volatility_score = self._calculate_volatility_score(df)
            volume_score = self._calculate_volume_score(df)
            
            # Weighted composite technical score
            technical_score = (
                trend_score * 0.35 + 
                momentum_score * 0.35 + 
                volatility_score * 0.15 + 
                volume_score * 0.15
            )
            
            # Determine signal type
            signal_type = self._determine_signal_type(technical_score, df)
            
            # Calculate confidence and risk
            confidence = self._calculate_confidence(df, technical_score)
            risk_score = self._calculate_technical_risk(df)
            
            # Generate reasons
            reasons = self._generate_technical_reasons(df, signal_type)
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                timestamp=latest.name,
                signal_type=signal_type,
                technical_score=technical_score,
                composite_score=technical_score,  # Will be updated with other analyses
                confidence_level=confidence,
                risk_score=risk_score,
                reasons=reasons,
                target_price=self._calculate_target_price(df, signal_type),
                stop_loss=self._calculate_stop_loss(df, signal_type),
                take_profit=self._calculate_take_profit(df, signal_type)
            )
            
            signals.append(signal)
            logger.debug(f"Generated {signal_type.value} signal for {symbol} with score {technical_score:.2f}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return []
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate trend-based score (0-100)"""
        try:
            latest = df.iloc[-1]
            score = 0
            max_score = 100
            
            # Price vs Moving Averages (30 points)
            if 'SMA_20' in df.columns and not pd.isna(latest['SMA_20']):
                if latest['Close'] > latest['SMA_20']:
                    score += 10
                if 'SMA_50' in df.columns and not pd.isna(latest['SMA_50']):
                    if latest['Close'] > latest['SMA_50']:
                        score += 10
                    if latest['SMA_20'] > latest['SMA_50']:
                        score += 10
            
            # MACD (25 points)
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
                    if latest['MACD'] > latest['MACD_Signal']:
                        score += 15
                    if latest['MACD'] > 0:
                        score += 10
            
            # ADX (25 points)
            if 'ADX' in df.columns and not pd.isna(latest['ADX']):
                if latest['ADX'] > 25:  # Strong trend
                    score += 15
                    if 'DI_Plus' in df.columns and 'DI_Minus' in df.columns:
                        if not pd.isna(latest['DI_Plus']) and not pd.isna(latest['DI_Minus']):
                            if latest['DI_Plus'] > latest['DI_Minus']:
                                score += 10
            
            # EMA alignment (20 points)
            if all(col in df.columns for col in ['EMA_12', 'EMA_26', 'EMA_50']):
                if (not pd.isna(latest['EMA_12']) and not pd.isna(latest['EMA_26']) and 
                    not pd.isna(latest['EMA_50'])):
                    if latest['EMA_12'] > latest['EMA_26'] > latest['EMA_50']:
                        score += 20
            
            return min(score, max_score)
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {str(e)}")
            return 50
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum-based score (0-100)"""
        try:
            latest = df.iloc[-1]
            score = 0
            max_score = 100
            
            # RSI (30 points)
            if 'RSI' in df.columns and not pd.isna(latest['RSI']):
                rsi = latest['RSI']
                if 30 < rsi < 70:  # Not overbought/oversold
                    score += 10
                if rsi > 50:  # Bullish momentum
                    score += 10
                if rsi > 60:  # Strong bullish momentum
                    score += 10
            
            # Stochastic (25 points)
            if 'STOCH_K' in df.columns and 'STOCH_D' in df.columns:
                if not pd.isna(latest['STOCH_K']) and not pd.isna(latest['STOCH_D']):
                    if latest['STOCH_K'] > latest['STOCH_D']:
                        score += 15
                    if latest['STOCH_K'] > 20 and latest['STOCH_K'] < 80:
                        score += 10
            
            # CCI (20 points)
            if 'CCI' in df.columns and not pd.isna(latest['CCI']):
                cci = latest['CCI']
                if -100 < cci < 100:
                    score += 10
                if cci > 0:
                    score += 10
            
            # Williams %R (25 points)
            if 'Williams_R' in df.columns and not pd.isna(latest['Williams_R']):
                wr = latest['Williams_R']
                if -80 < wr < -20:
                    score += 15
                if wr > -50:
                    score += 10
            
            return min(score, max_score)
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {str(e)}")
            return 50
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Calculate volatility-based score (0-100)"""
        try:
            latest = df.iloc[-1]
            score = 50  # Base score
            
            # Bollinger Bands position (40 points)
            if 'BB_Position' in df.columns and not pd.isna(latest['BB_Position']):
                bb_pos = latest['BB_Position']
                if 0.2 < bb_pos < 0.8:  # Not at extremes
                    score += 20
                if bb_pos > 0.5:  # Above middle
                    score += 20
            
            # ATR trend (30 points)
            if 'ATR' in df.columns and len(df) > 20:
                current_atr = df['ATR'].iloc[-1:].mean()
                past_atr = df['ATR'].iloc[-20:-1].mean()
                if not pd.isna(current_atr) and not pd.isna(past_atr):
                    if current_atr < past_atr:  # Decreasing volatility is good
                        score += 30
            
            # Historical Volatility (30 points)
            if 'HV_20' in df.columns and not pd.isna(latest['HV_20']):
                hv = latest['HV_20']
                if hv < 30:  # Low volatility
                    score += 30
                elif hv < 50:  # Moderate volatility
                    score += 15
            
            return min(score, 100)
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {str(e)}")
            return 50
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """Calculate volume-based score (0-100)"""
        try:
            latest = df.iloc[-1]
            score = 0
            
            # Volume confirmation (40 points)
            if 'Volume_Ratio' in df.columns and not pd.isna(latest['Volume_Ratio']):
                vol_ratio = latest['Volume_Ratio']
                if vol_ratio > 1.2:  # Above average volume
                    score += 40
                elif vol_ratio > 1.0:
                    score += 20
            
            # OBV trend (30 points)
            if 'OBV' in df.columns and len(df) > 5:
                obv_current = df['OBV'].iloc[-1]
                obv_past = df['OBV'].iloc[-6:-1].mean()
                if not pd.isna(obv_current) and not pd.isna(obv_past):
                    if obv_current > obv_past:
                        score += 30
            
            # MFI (30 points)
            if 'MFI' in df.columns and not pd.isna(latest['MFI']):
                mfi = latest['MFI']
                if 20 < mfi < 80:
                    score += 15
                if mfi > 50:
                    score += 15
            
            return min(score, 100)
            
        except Exception as e:
            logger.error(f"Error calculating volume score: {str(e)}")
            return 50
    
    def _determine_signal_type(self, technical_score: float, df: pd.DataFrame) -> SignalType:
        """Determine signal type based on technical score and conditions"""
        try:
            latest = df.iloc[-1]
            
            # Strong signals
            if technical_score >= 75:
                return SignalType.BUY
            elif technical_score <= 25:
                return SignalType.SELL
            else:
                # Check for specific patterns
                if self._check_bullish_patterns(df):
                    return SignalType.BUY
                elif self._check_bearish_patterns(df):
                    return SignalType.SELL
                else:
                    return SignalType.HOLD
                    
        except Exception as e:
            logger.error(f"Error determining signal type: {str(e)}")
            return SignalType.HOLD
    
    def _check_bullish_patterns(self, df: pd.DataFrame) -> bool:
        """Check for bullish technical patterns"""
        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            bullish_signals = 0
            
            # MACD crossover
            if ('MACD' in df.columns and 'MACD_Signal' in df.columns):
                if (latest['MACD'] > latest['MACD_Signal'] and 
                    previous['MACD'] <= previous['MACD_Signal']):
                    bullish_signals += 1
            
            # Price breaks above SMA
            if 'SMA_20' in df.columns:
                if (latest['Close'] > latest['SMA_20'] and 
                    previous['Close'] <= previous['SMA_20']):
                    bullish_signals += 1
            
            # RSI oversold recovery
            if 'RSI' in df.columns:
                if latest['RSI'] > 30 and previous['RSI'] <= 30:
                    bullish_signals += 1
            
            return bullish_signals >= 2
            
        except Exception as e:
            logger.error(f"Error checking bullish patterns: {str(e)}")
            return False
    
    def _check_bearish_patterns(self, df: pd.DataFrame) -> bool:
        """Check for bearish technical patterns"""
        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            bearish_signals = 0
            
            # MACD bearish crossover
            if ('MACD' in df.columns and 'MACD_Signal' in df.columns):
                if (latest['MACD'] < latest['MACD_Signal'] and 
                    previous['MACD'] >= previous['MACD_Signal']):
                    bearish_signals += 1
            
            # Price breaks below SMA
            if 'SMA_20' in df.columns:
                if (latest['Close'] < latest['SMA_20'] and 
                    previous['Close'] >= previous['SMA_20']):
                    bearish_signals += 1
            
            # RSI overbought decline
            if 'RSI' in df.columns:
                if latest['RSI'] < 70 and previous['RSI'] >= 70:
                    bearish_signals += 1
            
            return bearish_signals >= 2
            
        except Exception as e:
            logger.error(f"Error checking bearish patterns: {str(e)}")
            return False
    
    # Manual calculation methods (when TA-Lib is not available)
    def _calculate_rsi_manual(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Manual RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_adx_manual(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Manual ADX calculation"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate directional movements
        dm_plus = high - high.shift()
        dm_minus = low.shift() - low
        
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        
        # Calculate smoothed averages
        atr = tr.rolling(window=period).mean()
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx, di_plus, di_minus
    
    def _calculate_stochastic_manual(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Manual Stochastic calculation"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def _calculate_cci_manual(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Manual CCI calculation"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        return (tp - ma) / (0.015 * md)
    
    def _calculate_williams_r_manual(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Manual Williams %R calculation"""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        return -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    
    def _calculate_atr_manual(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Manual ATR calculation"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()
    
    def _calculate_obv_manual(self, df: pd.DataFrame) -> pd.Series:
        """Manual OBV calculation"""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    def _calculate_mfi_manual(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Manual Money Flow Index calculation"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_mf = []
        negative_mf = []
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_mf.append(money_flow.iloc[i])
                negative_mf.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_mf.append(0)
                negative_mf.append(money_flow.iloc[i])
            else:
                positive_mf.append(0)
                negative_mf.append(0)
        
        positive_mf = pd.Series([0] + positive_mf, index=df.index)
        negative_mf = pd.Series([0] + negative_mf, index=df.index)
        
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        mfr = positive_mf_sum / negative_mf_sum
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi
    
    def _calculate_confidence(self, df: pd.DataFrame, technical_score: float) -> float:
        """Calculate confidence level for the signal"""
        try:
            confidence = 50  # Base confidence
            
            # Score-based confidence
            if technical_score > 80 or technical_score < 20:
                confidence += 30
            elif technical_score > 70 or technical_score < 30:
                confidence += 20
            elif technical_score > 60 or technical_score < 40:
                confidence += 10
            
            # Volume confirmation
            if 'Volume_Ratio' in df.columns:
                latest_vol_ratio = df['Volume_Ratio'].iloc[-1]
                if not pd.isna(latest_vol_ratio) and latest_vol_ratio > 1.5:
                    confidence += 15
            
            # Trend strength
            if 'ADX' in df.columns:
                latest_adx = df['ADX'].iloc[-1]
                if not pd.isna(latest_adx) and latest_adx > 25:
                    confidence += 15
            
            return min(confidence, 100)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 50
    
    def _calculate_technical_risk(self, df: pd.DataFrame) -> float:
        """Calculate technical risk score"""
        try:
            risk = 50  # Base risk
            
            # Volatility risk
            if 'ATR' in df.columns:
                current_atr = df['ATR'].iloc[-1]
                if not pd.isna(current_atr):
                    avg_price = df['Close'].iloc[-20:].mean()
                    atr_percent = (current_atr / avg_price) * 100
                    if atr_percent > 5:
                        risk += 20
                    elif atr_percent > 3:
                        risk += 10
            
            # RSI extremes
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if not pd.isna(rsi):
                    if rsi > 80 or rsi < 20:
                        risk += 20
            
            # Price vs Bollinger Bands
            if 'BB_Position' in df.columns:
                bb_pos = df['BB_Position'].iloc[-1]
                if not pd.isna(bb_pos):
                    if bb_pos > 0.9 or bb_pos < 0.1:
                        risk += 15
            
            return min(risk, 100)
            
        except Exception as e:
            logger.error(f"Error calculating technical risk: {str(e)}")
            return 50
    
    def _generate_technical_reasons(self, df: pd.DataFrame, signal_type: SignalType) -> List[str]:
        """Generate reasons for the technical signal"""
        reasons = []
        latest = df.iloc[-1]
        
        try:
            if signal_type == SignalType.BUY:
                if 'SMA_20' in df.columns and latest['Close'] > latest['SMA_20']:
                    reasons.append("Price above 20-day SMA")
                if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                    if latest['MACD'] > latest['MACD_Signal']:
                        reasons.append("MACD bullish crossover")
                if 'RSI' in df.columns and 30 < latest['RSI'] < 70:
                    reasons.append("RSI in healthy range")
                if 'Volume_Ratio' in df.columns and latest['Volume_Ratio'] > 1.2:
                    reasons.append("Above average volume")
                    
            elif signal_type == SignalType.SELL:
                if 'SMA_20' in df.columns and latest['Close'] < latest['SMA_20']:
                    reasons.append("Price below 20-day SMA")
                if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                    if latest['MACD'] < latest['MACD_Signal']:
                        reasons.append("MACD bearish crossover")
                if 'RSI' in df.columns and latest['RSI'] > 70:
                    reasons.append("RSI overbought")
            
            else:  # HOLD
                reasons.append("Mixed technical signals")
                if 'RSI' in df.columns and 45 < latest['RSI'] < 55:
                    reasons.append("RSI neutral")
                    
        except Exception as e:
            logger.error(f"Error generating technical reasons: {str(e)}")
            reasons.append("Technical analysis completed")
        
        return reasons[:5]  # Limit to 5 reasons
    
    def _calculate_target_price(self, df: pd.DataFrame, signal_type: SignalType) -> Optional[float]:
        """Calculate target price based on technical levels"""
        try:
            latest = df.iloc[-1]
            current_price = latest['Close']
            
            if signal_type == SignalType.BUY:
                # Use resistance levels or ATR-based targets
                if 'BB_Upper' in df.columns and not pd.isna(latest['BB_Upper']):
                    return latest['BB_Upper']
                elif 'ATR' in df.columns and not pd.isna(latest['ATR']):
                    return current_price + (latest['ATR'] * 2)
                else:
                    return current_price * 1.05  # 5% target
                    
            elif signal_type == SignalType.SELL:
                # Use support levels
                if 'BB_Lower' in df.columns and not pd.isna(latest['BB_Lower']):
                    return latest['BB_Lower']
                elif 'ATR' in df.columns and not pd.isna(latest['ATR']):
                    return current_price - (latest['ATR'] * 2)
                else:
                    return current_price * 0.95  # 5% target
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating target price: {str(e)}")
            return None
    
    def _calculate_stop_loss(self, df: pd.DataFrame, signal_type: SignalType) -> Optional[float]:
        """Calculate stop loss based on technical levels"""
        try:
            latest = df.iloc[-1]
            current_price = latest['Close']
            
            if signal_type == SignalType.BUY:
                # Use support levels or ATR-based stops
                if 'ATR' in df.columns and not pd.isna(latest['ATR']):
                    return current_price - (latest['ATR'] * 1.5)
                else:
                    return current_price * 0.98  # 2% stop
                    
            elif signal_type == SignalType.SELL:
                # Use resistance levels
                if 'ATR' in df.columns and not pd.isna(latest['ATR']):
                    return current_price + (latest['ATR'] * 1.5)
                else:
                    return current_price * 1.02  # 2% stop
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return None
    
    def _calculate_take_profit(self, df: pd.DataFrame, signal_type: SignalType) -> Optional[float]:
        """Calculate take profit level"""
        try:
            latest = df.iloc[-1]
            current_price = latest['Close']
            
            if signal_type == SignalType.BUY:
                if 'ATR' in df.columns and not pd.isna(latest['ATR']):
                    return current_price + (latest['ATR'] * 3)
                else:
                    return current_price * 1.06  # 6% profit
                    
            elif signal_type == SignalType.SELL:
                if 'ATR' in df.columns and not pd.isna(latest['ATR']):
                    return current_price - (latest['ATR'] * 3)
                else:
                    return current_price * 0.94  # 6% profit
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return None

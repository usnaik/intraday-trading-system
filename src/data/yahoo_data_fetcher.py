"""
Yahoo Finance data fetcher for real-time and historical market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
from loguru import logger

from ..models.data_models import StockData, MarketSegment


class YahooDataFetcher:
    """Fetches market data from Yahoo Finance"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_stock_data(
        self, 
        symbols: List[str], 
        period: str = "1d", 
        interval: str = "1m"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        try:
            logger.info(f"Fetching data for symbols: {symbols}")
            
            data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period, interval=interval)
                    
                    if not hist.empty:
                        # Clean and prepare data
                        hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
                        hist = hist.dropna()
                        
                        # Add symbol column
                        hist['Symbol'] = symbol
                        
                        data[symbol] = hist
                        logger.debug(f"Successfully fetched {len(hist)} records for {symbol}")
                    else:
                        logger.warning(f"No data found for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    continue
                    
            logger.info(f"Successfully fetched data for {len(data)} symbols")
            return data
            
        except Exception as e:
            logger.error(f"Error in get_stock_data: {str(e)}")
            return {}
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get real-time/current data for symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with current market data
        """
        try:
            logger.info(f"Fetching real-time data for: {symbols}")
            
            data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Get current price and basic info
                    current_data = {
                        'symbol': symbol,
                        'current_price': info.get('regularMarketPrice', 0),
                        'previous_close': info.get('previousClose', 0),
                        'open': info.get('regularMarketOpen', 0),
                        'day_high': info.get('dayHigh', 0),
                        'day_low': info.get('dayLow', 0),
                        'volume': info.get('regularMarketVolume', 0),
                        'market_cap': info.get('marketCap', 0),
                        'timestamp': datetime.now()
                    }
                    
                    data[symbol] = current_data
                    
                except Exception as e:
                    logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
                    continue
                    
            return data
            
        except Exception as e:
            logger.error(f"Error in get_real_time_data: {str(e)}")
            return {}
    
    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get fundamental data for symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with fundamental metrics
        """
        try:
            logger.info(f"Fetching fundamental data for: {symbols}")
            
            data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Extract fundamental metrics
                    fundamentals = {
                        'symbol': symbol,
                        'pe_ratio': info.get('forwardPE', info.get('trailingPE')),
                        'pb_ratio': info.get('priceToBook'),
                        'roe': info.get('returnOnEquity'),
                        'roa': info.get('returnOnAssets'),
                        'profit_margin': info.get('profitMargins'),
                        'operating_margin': info.get('operatingMargins'),
                        'debt_to_equity': info.get('debtToEquity'),
                        'current_ratio': info.get('currentRatio'),
                        'quick_ratio': info.get('quickRatio'),
                        'revenue_growth': info.get('revenueGrowth'),
                        'earnings_growth': info.get('earningsGrowth'),
                        'market_cap': info.get('marketCap'),
                        'beta': info.get('beta'),
                        'ev_ebitda': info.get('enterpriseToEbitda'),
                        'timestamp': datetime.now()
                    }
                    
                    data[symbol] = fundamentals
                    
                except Exception as e:
                    logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
                    continue
                    
            return data
            
        except Exception as e:
            logger.error(f"Error in get_fundamental_data: {str(e)}")
            return {}
    
    def get_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for backtesting
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for historical data
            end_date: End date (default: today)
            interval: Data interval
            
        Returns:
            Dictionary with historical data
        """
        if end_date is None:
            end_date = datetime.now()
            
        try:
            logger.info(f"Fetching historical data from {start_date} to {end_date}")
            
            data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date, interval=interval)
                    
                    if not hist.empty:
                        hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
                        hist = hist.dropna()
                        hist['Symbol'] = symbol
                        data[symbol] = hist
                        
                except Exception as e:
                    logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
                    continue
                    
            return data
            
        except Exception as e:
            logger.error(f"Error in get_historical_data: {str(e)}")
            return {}
    
    def get_options_data(self, symbol: str) -> Dict:
        """
        Get options data for a symbol (for volatility analysis)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Options data dictionary
        """
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            
            if not options_dates:
                return {}
                
            # Get options for nearest expiration
            options = ticker.option_chain(options_dates[0])
            
            return {
                'calls': options.calls.to_dict('records'),
                'puts': options.puts.to_dict('records'),
                'expiration': options_dates[0]
            }
            
        except Exception as e:
            logger.error(f"Error fetching options data for {symbol}: {str(e)}")
            return {}
    
    def get_earnings_calendar(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get upcoming earnings dates (basic implementation)
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with earnings information
        """
        try:
            earnings_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    calendar = ticker.calendar
                    
                    if calendar is not None and not calendar.empty:
                        earnings_data[symbol] = {
                            'next_earnings': calendar.index[0] if len(calendar.index) > 0 else None,
                            'earnings_data': calendar.to_dict('records')
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not fetch earnings calendar for {symbol}: {str(e)}")
                    continue
                    
            return earnings_data
            
        except Exception as e:
            logger.error(f"Error in get_earnings_calendar: {str(e)}")
            return {}
    
    def get_sector_performance(self) -> Dict[str, float]:
        """
        Get sector performance using sector ETFs as proxies
        
        Returns:
            Dictionary with sector performance
        """
        sector_etfs = {
            'Technology': 'XLK',
            'Finance': 'XLF', 
            'Healthcare': 'XLV',
            'Energy': 'XLE',
            'Consumer Goods': 'XLP',
            'Retail': 'XLY',
            'Telecommunications': 'XTL',
            'Industrials': 'XLI',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE'
        }
        
        try:
            performance = {}
            
            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        previous_close = hist['Open'].iloc[0]
                        change_pct = ((current_price - previous_close) / previous_close) * 100
                        performance[sector] = change_pct
                        
                except Exception as e:
                    logger.warning(f"Could not fetch performance for {sector} ({etf}): {str(e)}")
                    continue
                    
            return performance
            
        except Exception as e:
            logger.error(f"Error in get_sector_performance: {str(e)}")
            return {}
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate if symbols exist and are tradeable
        
        Args:
            symbols: List of stock symbols to validate
            
        Returns:
            List of valid symbols
        """
        try:
            valid_symbols = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Check if symbol has basic required data
                    if info.get('regularMarketPrice') is not None:
                        valid_symbols.append(symbol)
                    else:
                        logger.warning(f"Symbol {symbol} appears to be invalid or not tradeable")
                        
                except Exception as e:
                    logger.warning(f"Could not validate symbol {symbol}: {str(e)}")
                    continue
                    
            logger.info(f"Validated {len(valid_symbols)} out of {len(symbols)} symbols")
            return valid_symbols
            
        except Exception as e:
            logger.error(f"Error in validate_symbols: {str(e)}")
            return symbols  # Return original list if validation fails


# Utility functions
def format_stock_data_for_model(df: pd.DataFrame, symbol: str, segment: MarketSegment) -> List[StockData]:
    """Convert DataFrame to StockData models"""
    stock_data_list = []
    
    for index, row in df.iterrows():
        stock_data = StockData(
            symbol=symbol,
            timestamp=index,
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=row['Volume'],
            segment=segment
        )
        stock_data_list.append(stock_data)
    
    return stock_data_list


def calculate_basic_statistics(df: pd.DataFrame) -> Dict:
    """Calculate basic statistical measures for a stock"""
    try:
        returns = df['Close'].pct_change().dropna()
        
        stats = {
            'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': ((df['Close'] / df['Close'].expanding().max()) - 1).min(),
            'beta': calculate_beta(df, '^GSPC'),  # Beta vs S&P 500
            'correlation_market': returns.corrwith(get_market_returns())
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        return {}


def calculate_beta(stock_df: pd.DataFrame, market_symbol: str = '^GSPC') -> float:
    """Calculate beta coefficient"""
    try:
        # Get market data
        market_ticker = yf.Ticker(market_symbol)
        market_data = market_ticker.history(period="1y")
        
        # Calculate returns
        stock_returns = stock_df['Close'].pct_change().dropna()
        market_returns = market_data['Close'].pct_change().dropna()
        
        # Align dates
        common_dates = stock_returns.index.intersection(market_returns.index)
        stock_aligned = stock_returns.loc[common_dates]
        market_aligned = market_returns.loc[common_dates]
        
        # Calculate beta
        covariance = np.cov(stock_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        beta = covariance / market_variance if market_variance != 0 else 1.0
        return beta
        
    except Exception as e:
        logger.error(f"Error calculating beta: {str(e)}")
        return 1.0


def get_market_returns(period: str = "1y") -> pd.Series:
    """Get market returns (S&P 500)"""
    try:
        market_ticker = yf.Ticker('^GSPC')
        market_data = market_ticker.history(period=period)
        return market_data['Close'].pct_change().dropna()
    except Exception as e:
        logger.error(f"Error getting market returns: {str(e)}")
        return pd.Series()

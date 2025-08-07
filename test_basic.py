#!/usr/bin/env python3
"""
Basic test to verify the trading system works
"""

import sys
import os
import asyncio
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.core.trading_system import IntradayTradingSystem
    from src.data.yahoo_data_fetcher import YahooDataFetcher
    from src.technical.technical_analyzer import TechnicalAnalyzer
    from src.fundamental.fundamental_analyzer import FundamentalAnalyzer
    
    print("✅ All imports successful!")
    
    # Test data fetcher
    print("\n🔍 Testing data fetcher...")
    data_fetcher = YahooDataFetcher()
    test_symbols = ['AAPL', 'MSFT']
    
    # Validate symbols
    valid_symbols = data_fetcher.validate_symbols(test_symbols)
    print(f"✅ Validated symbols: {valid_symbols}")
    
    # Test basic data fetching
    stock_data = data_fetcher.get_stock_data(valid_symbols[:1], period="1d", interval="1h")
    if stock_data:
        symbol = list(stock_data.keys())[0]
        df = stock_data[symbol]
        print(f"✅ Fetched {len(df)} data points for {symbol}")
        print(f"   Latest close price: ${df['Close'].iloc[-1]:.2f}")
    else:
        print("⚠️  No stock data returned")
    
    # Test technical analyzer
    print("\n📈 Testing technical analyzer...")
    if stock_data:
        tech_analyzer = TechnicalAnalyzer()
        symbol = list(stock_data.keys())[0]
        df_with_indicators = tech_analyzer.calculate_all_indicators(stock_data[symbol])
        print(f"✅ Calculated technical indicators")
        
        # Check some indicators
        if 'RSI' in df_with_indicators.columns:
            latest_rsi = df_with_indicators['RSI'].iloc[-1]
            if not pd.isna(latest_rsi):
                print(f"   Latest RSI: {latest_rsi:.2f}")
        
        if 'MACD' in df_with_indicators.columns:
            latest_macd = df_with_indicators['MACD'].iloc[-1]
            if not pd.isna(latest_macd):
                print(f"   Latest MACD: {latest_macd:.4f}")
    
    # Test fundamental analyzer
    print("\n📊 Testing fundamental analyzer...")
    fund_analyzer = FundamentalAnalyzer()
    fundamental_data = await fund_analyzer.analyze_fundamentals(valid_symbols[:1])
    if fundamental_data:
        symbol = list(fundamental_data.keys())[0]
        metrics = fundamental_data[symbol]
        print(f"✅ Fetched fundamental data for {symbol}")
        if metrics.pe_ratio:
            print(f"   P/E Ratio: {metrics.pe_ratio:.2f}")
    
    print("\n🎉 Basic tests completed successfully!")
    print("\nYou can now run the full system with:")
    print("python main.py --symbols AAPL MSFT")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install required dependencies:")
    print("pip install -r requirements.txt")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    print("\nCheck the logs for more details.")

if __name__ == "__main__":
    # Run async components
    import pandas as pd
    asyncio.run(main()) if 'main' in locals() else None

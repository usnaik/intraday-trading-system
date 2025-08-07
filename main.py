#!/usr/bin/env python3
"""
Main entry point for the Intraday Trading System
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from src.core.trading_system import IntradayTradingSystem, run_trading_system


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File logging
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger.add(
        os.path.join(log_dir, "trading_system.log"),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="30 days",
        compression="zip"
    )


async def run_analysis_mode(config_path: str = None, symbols: list = None):
    """Run one-time analysis"""
    logger.info("Starting one-time analysis mode")
    
    system = IntradayTradingSystem(config_path)
    
    try:
        # Run analysis
        signals = await system.run_analysis(symbols)
        
        # Display results
        print("\n" + "="*60)
        print(" INTRADAY TRADING SYSTEM - ANALYSIS RESULTS")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total signals generated: {len(signals)}")
        
        if signals:
            # Separate signals by type
            buy_signals = [s for s in signals.values() if s.signal_type.value == 'BUY']
            sell_signals = [s for s in signals.values() if s.signal_type.value == 'SELL']
            hold_signals = [s for s in signals.values() if s.signal_type.value == 'HOLD']
            
            print(f"\nSignal Breakdown:")
            print(f"  • BUY signals: {len(buy_signals)}")
            print(f"  • SELL signals: {len(sell_signals)}")
            print(f"  • HOLD signals: {len(hold_signals)}")
            
            # Display top BUY signals
            if buy_signals:
                print(f"\n🟢 TOP BUY SIGNALS:")
                print("-" * 50)
                buy_signals.sort(key=lambda x: x.composite_score, reverse=True)
                
                for i, signal in enumerate(buy_signals[:5], 1):
                    print(f"{i}. {signal.symbol}")
                    print(f"   Score: {signal.composite_score:.1f}/100 | Confidence: {signal.confidence_level:.1f}% | Risk: {signal.risk_score:.1f}/100")
                    if signal.reasons:
                        print(f"   Reasons: {'; '.join(signal.reasons[:3])}")
                    if signal.target_price:
                        print(f"   Target: ${signal.target_price:.2f} | Stop Loss: ${signal.stop_loss:.2f}")
                    print()
            
            # Display top SELL signals
            if sell_signals:
                print(f"\n🔴 TOP SELL SIGNALS:")
                print("-" * 50)
                sell_signals.sort(key=lambda x: x.composite_score)
                
                for i, signal in enumerate(sell_signals[:5], 1):
                    print(f"{i}. {signal.symbol}")
                    print(f"   Score: {signal.composite_score:.1f}/100 | Confidence: {signal.confidence_level:.1f}% | Risk: {signal.risk_score:.1f}/100")
                    if signal.reasons:
                        print(f"   Reasons: {'; '.join(signal.reasons[:3])}")
                    print()
        
        # Display market overview
        market_overview = system.get_market_overview()
        if market_overview.get('sector_performance'):
            print(f"\n📊 SECTOR PERFORMANCE:")
            print("-" * 30)
            for sector, performance in market_overview['sector_performance'].items():
                emoji = "🟢" if performance > 0 else "🔴" if performance < 0 else "⚪"
                print(f"{emoji} {sector}: {performance:+.2f}%")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error in analysis mode: {str(e)}")
        return 1
    finally:
        await system.shutdown()
    
    return 0


async def run_monitor_mode(config_path: str = None):
    """Run continuous monitoring mode"""
    logger.info("Starting continuous monitoring mode")
    
    system = IntradayTradingSystem(config_path)
    
    try:
        await system.start_realtime_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring mode: {str(e)}")
        return 1
    finally:
        await system.shutdown()
    
    return 0


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Intraday Trading System - AI-powered stock analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run one-time analysis
  %(prog)s --monitor                 # Run continuous monitoring
  %(prog)s --symbols AAPL MSFT GOOGL # Analyze specific stocks
  %(prog)s --config custom_config.yaml # Use custom config
  %(prog)s --log-level DEBUG         # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--monitor', '-m',
        action='store_true',
        help='Run in continuous monitoring mode'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        help='Specific symbols to analyze (space-separated)'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Intraday Trading System v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("Intraday Trading System starting...")
    
    try:
        # Check if config file exists
        if args.config and not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            return 1
        
        # Run appropriate mode
        if args.monitor:
            exit_code = asyncio.run(run_monitor_mode(args.config))
        else:
            exit_code = asyncio.run(run_analysis_mode(args.config, args.symbols))
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

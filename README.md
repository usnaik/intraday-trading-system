# Intraday Trading System

An AI-powered intraday trading system that combines **Fundamental**, **Technical**, and **Sentiment Analysis** to generate trading recommendations for stocks across 10 market segments.

## 🚀 Features

### Multi-Analysis Approach
- **📊 Fundamental Analysis**: Company financial health, ratios, and growth metrics
- **📈 Technical Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands, ADX
- **🔍 Sentiment Analysis**: Twitter, news, and Reddit sentiment using VADER, TextBlob, and FinBERT

### Market Coverage
- **10 Market Segments**: Technology, Finance, Healthcare, Energy, Consumer Goods, Retail, Telecommunications, Industrials, Utilities, Real Estate  
- **5 Stocks per Segment**: 50 total stocks for comprehensive coverage

### Risk Management
- Position sizing based on signal strength and portfolio risk
- Correlation-based diversification
- Daily loss limits and stop-loss management
- Portfolio risk metrics and suggestions

### Data Sources
- **Yahoo Finance**: Real-time and historical stock data
- **Twitter API**: Social media sentiment
- **News API**: Financial news sentiment  
- **Reddit API**: Community discussions
- **Alpha Vantage**: Additional market data (optional)

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection for data fetching
- API keys for sentiment analysis (optional but recommended)

## 🛠 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/intraday-trading-system.git
cd intraday-trading-system
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install TA-Lib (optional but recommended):**
```bash
# On Ubuntu/Debian:
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On macOS:
brew install ta-lib
pip install TA-Lib

# On Windows:
# Download TA-Lib from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

5. **Set up configuration:**
```bash
cp .env.example .env
# Edit .env with your API keys (optional)
```

## ⚙️ Configuration

### API Keys (Optional)
Edit the `.env` file to add your API keys for enhanced functionality:

```bash
# Twitter API (for sentiment analysis)
TWITTER_BEARER_TOKEN=your_token_here
TWITTER_API_KEY=your_api_key_here
TWITTER_API_SECRET=your_api_secret_here

# News API
NEWS_API_KEY=your_news_api_key_here

# Reddit API  
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here

# Alpha Vantage API
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

### System Configuration
The system uses `config/config.yaml` for configuration. Key settings:

- **Market Segments**: Define stocks for each segment
- **Analysis Weights**: Adjust fundamental/technical/sentiment weights
- **Risk Management**: Set position limits and risk parameters
- **Trading Parameters**: Configure stop-loss, take-profit levels

## 🚀 Usage

### One-time Analysis
Run analysis for all configured stocks:
```bash
python main.py
```

Analyze specific stocks:
```bash
python main.py --symbols AAPL MSFT GOOGL
```

### Continuous Monitoring
Start real-time monitoring mode:
```bash
python main.py --monitor
```

### Custom Configuration
Use a custom configuration file:
```bash
python main.py --config my_config.yaml
```

### Debug Mode
Enable detailed logging:
```bash
python main.py --log-level DEBUG
```

## 📊 Sample Output

```
============================================================
 INTRADAY TRADING SYSTEM - ANALYSIS RESULTS
============================================================
Timestamp: 2025-01-07 13:05:45
Total signals generated: 15

Signal Breakdown:
  • BUY signals: 5
  • SELL signals: 2  
  • HOLD signals: 8

🟢 TOP BUY SIGNALS:
--------------------------------------------------
1. NVDA
   Score: 85.3/100 | Confidence: 78.5% | Risk: 45.2/100
   Reasons: Strong buy recommendation; Technical: Price above 20-day SMA; Fundamental: Strong ROE of 35.2%
   Target: $890.50 | Stop Loss: $820.30

2. MSFT
   Score: 82.1/100 | Confidence: 72.3% | Risk: 38.7/100
   Reasons: Strong buy recommendation; Technical: MACD bullish crossover; Sentiment: Positive social media sentiment

📊 SECTOR PERFORMANCE:
------------------------------
🟢 Technology: +1.45%
🔴 Finance: -0.32%
🟢 Healthcare: +0.78%
============================================================
```

## 🏗 System Architecture

```
intraday-trading-system/
├── src/
│   ├── core/               # Main trading system
│   ├── data/               # Data fetchers (Yahoo Finance)
│   ├── technical/          # Technical analysis
│   ├── fundamental/        # Fundamental analysis  
│   ├── sentiment/          # Sentiment analysis
│   ├── models/             # Data models
│   └── utils/              # Database & risk management
├── config/                 # Configuration files
├── data/                   # Database storage
├── tests/                  # Unit tests
└── logs/                   # Application logs
```

### Key Components

1. **TradingSystem**: Orchestrates all analyses
2. **DataFetcher**: Retrieves market data from Yahoo Finance
3. **TechnicalAnalyzer**: Calculates 20+ technical indicators
4. **FundamentalAnalyzer**: Evaluates company financial metrics  
5. **SentimentAnalyzer**: Processes social media sentiment
6. **RiskManager**: Implements position sizing and risk controls
7. **DatabaseManager**: Handles data persistence

## 📈 Analysis Details

### Technical Indicators
- **Trend**: SMA, EMA, MACD, ADX, Parabolic SAR
- **Momentum**: RSI, Stochastic, CCI, Williams %R
- **Volatility**: Bollinger Bands, ATR, Historical Volatility
- **Volume**: OBV, VWAP, MFI, Volume ratios

### Fundamental Metrics
- **Profitability**: ROE, ROA, Profit Margin, Operating Margin
- **Valuation**: P/E, P/B, EV/EBITDA ratios
- **Growth**: Revenue Growth, Earnings Growth
- **Financial Health**: Debt/Equity, Current Ratio, Quick Ratio

### Sentiment Sources
- **Twitter**: Real-time tweets and mentions
- **News**: Financial news headlines and articles
- **Reddit**: Investment community discussions
- **Models**: VADER, TextBlob, FinBERT for financial sentiment

## 🔧 Customization

### Adding New Stocks
Edit `config/config.yaml` to add stocks to market segments:

```yaml
market_segments:
  technology:
    stocks: ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA"]  # Add TSLA
```

### Adjusting Analysis Weights
Modify the relative importance of each analysis type:

```yaml
analysis_weights:
  fundamental: 0.25  # Reduce fundamental weight
  technical: 0.50    # Increase technical weight  
  sentiment: 0.25    # Keep sentiment weight
```

### Risk Management Settings
Configure position sizing and risk limits:

```yaml
trading:
  max_positions: 15           # Increase max positions
  position_size_percent: 8.0  # Reduce position size
  stop_loss_percent: 1.5      # Tighter stop loss
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_technical.py -v
pytest tests/test_fundamental.py -v
pytest tests/test_sentiment.py -v
```

## 📝 Logging

Logs are written to:
- **Console**: Real-time colored output
- **File**: `logs/trading_system.log` with rotation

Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## ⚠️ Disclaimers

- **Educational Purpose**: This system is for educational and research purposes only
- **Not Investment Advice**: Do not use for actual trading without proper testing
- **Risk Warning**: Trading involves risk of financial loss
- **Backtesting Required**: Test thoroughly before any live usage
- **API Limitations**: Respect rate limits of data providers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/intraday-trading-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/intraday-trading-system/discussions)

## 🙏 Acknowledgments

- Yahoo Finance for market data
- TA-Lib for technical analysis functions
- Various Python libraries that make this project possible
- The open-source community for inspiration and tools

---

**⚡ Happy Trading! Remember: Past performance does not guarantee future results.**

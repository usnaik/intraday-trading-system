"""
Sentiment Analysis Module for the Intraday Trading System
Analyzes sentiment from Twitter, news, and Reddit for trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
import asyncio
import aiohttp
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
import os

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logger.warning("Tweepy not available. Twitter sentiment analysis disabled.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. FinBERT sentiment analysis disabled.")

from ..models.data_models import SentimentData, SignalType


class SentimentAnalyzer:
    """Sentiment analysis engine for trading signals"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.sentiment_config = self.config.get('sentiment_analysis', {})
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize Twitter API if credentials are available
        self.twitter_api = None
        self._init_twitter_api()
        
        # Initialize FinBERT if available
        self.finbert_analyzer = None
        self._init_finbert()
        
        # News API configuration
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
        # Reddit API configuration  
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        # Sentiment keywords for each stock
        self.stock_keywords = {}
        
    def _init_twitter_api(self):
        """Initialize Twitter API client"""
        if not TWEEPY_AVAILABLE:
            return
            
        try:
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            api_key = os.getenv('TWITTER_API_KEY')
            api_secret = os.getenv('TWITTER_API_SECRET')
            access_token = os.getenv('TWITTER_ACCESS_TOKEN')
            access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            
            if all([bearer_token, api_key, api_secret, access_token, access_token_secret]):
                # Initialize Twitter API v2 client
                self.twitter_client = tweepy.Client(
                    bearer_token=bearer_token,
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret,
                    wait_on_rate_limit=True
                )
                
                # Initialize API v1.1 for additional features
                auth = tweepy.OAuthHandler(api_key, api_secret)
                auth.set_access_token(access_token, access_token_secret)
                self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
                
                logger.info("Twitter API initialized successfully")
            else:
                logger.warning("Twitter API credentials not found. Twitter sentiment disabled.")
                
        except Exception as e:
            logger.error(f"Error initializing Twitter API: {str(e)}")
    
    def _init_finbert(self):
        """Initialize FinBERT for financial sentiment analysis"""
        if not TRANSFORMERS_AVAILABLE:
            return
            
        try:
            # Use FinBERT model specifically trained on financial data
            self.finbert_analyzer = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
            logger.info("FinBERT sentiment analyzer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize FinBERT: {str(e)}")
    
    async def analyze_sentiment(self, symbols: List[str]) -> Dict[str, SentimentData]:
        """
        Analyze sentiment for multiple symbols from all sources
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with sentiment data for each symbol
        """
        logger.info(f"Analyzing sentiment for symbols: {symbols}")
        
        sentiment_results = {}
        
        # Process each symbol
        for symbol in symbols:
            try:
                # Generate keywords for the stock
                keywords = self._generate_stock_keywords(symbol)
                
                # Analyze sentiment from different sources concurrently
                tasks = []
                
                if self.twitter_client:
                    tasks.append(self._analyze_twitter_sentiment(symbol, keywords))
                else:
                    tasks.append(self._create_empty_twitter_sentiment())
                
                if self.news_api_key:
                    tasks.append(self._analyze_news_sentiment(symbol, keywords))
                else:
                    tasks.append(self._create_empty_news_sentiment())
                
                if self.reddit_client_id:
                    tasks.append(self._analyze_reddit_sentiment(symbol, keywords))
                else:
                    tasks.append(self._create_empty_reddit_sentiment())
                
                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                twitter_data = results[0] if not isinstance(results[0], Exception) else {}
                news_data = results[1] if not isinstance(results[1], Exception) else {}
                reddit_data = results[2] if not isinstance(results[2], Exception) else {}
                
                # Combine sentiment data
                sentiment_data = SentimentData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    twitter_sentiment=twitter_data.get('sentiment', 0.0),
                    news_sentiment=news_data.get('sentiment', 0.0),
                    reddit_sentiment=reddit_data.get('sentiment', 0.0),
                    twitter_mentions=twitter_data.get('mentions', 0),
                    news_mentions=news_data.get('mentions', 0),
                    reddit_mentions=reddit_data.get('mentions', 0)
                )
                
                # Calculate composite sentiment
                sentiment_data.composite_sentiment = self._calculate_composite_sentiment(sentiment_data)
                sentiment_data.sentiment_strength = self._calculate_sentiment_strength(sentiment_data)
                
                sentiment_results[symbol] = sentiment_data
                logger.debug(f"Sentiment analysis completed for {symbol}")
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
                # Create default sentiment data
                sentiment_results[symbol] = SentimentData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    composite_sentiment=0.0,
                    sentiment_strength=0.0
                )
        
        logger.info(f"Sentiment analysis completed for {len(sentiment_results)} symbols")
        return sentiment_results
    
    def _generate_stock_keywords(self, symbol: str) -> List[str]:
        """Generate search keywords for a stock symbol"""
        # Base keywords
        keywords = [
            f"${symbol}",  # Stock symbol with $
            symbol,       # Just the symbol
            f"{symbol} stock",
            f"{symbol} shares",
            f"{symbol} trading"
        ]
        
        # Add company-specific keywords based on symbol
        company_keywords = {
            'AAPL': ['Apple', 'iPhone', 'iPad', 'Mac', 'Tim Cook'],
            'MSFT': ['Microsoft', 'Windows', 'Azure', 'Office', 'Satya Nadella'],
            'GOOGL': ['Google', 'Alphabet', 'Search', 'YouTube', 'Android'],
            'META': ['Meta', 'Facebook', 'Instagram', 'WhatsApp', 'Metaverse'],
            'NVDA': ['Nvidia', 'GPU', 'AI chips', 'graphics cards'],
            'TSLA': ['Tesla', 'Elon Musk', 'electric vehicle', 'EV'],
            'AMZN': ['Amazon', 'AWS', 'Prime', 'Jeff Bezos'],
            'JPM': ['JPMorgan', 'Chase Bank', 'Jamie Dimon'],
            'JNJ': ['Johnson & Johnson', 'pharmaceutical', 'medical devices'],
            'PG': ['Procter & Gamble', 'consumer goods', 'household products']
        }
        
        if symbol in company_keywords:
            keywords.extend(company_keywords[symbol])
        
        return keywords[:10]  # Limit to 10 keywords
    
    async def _analyze_twitter_sentiment(self, symbol: str, keywords: List[str]) -> Dict:
        """Analyze Twitter sentiment for a symbol"""
        try:
            if not self.twitter_client:
                return {'sentiment': 0.0, 'mentions': 0}
            
            logger.debug(f"Fetching Twitter data for {symbol}")
            
            # Build search query
            query = f"({' OR '.join(keywords[:5])}) -is:retweet lang:en"
            
            # Fetch recent tweets (last 24 hours)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=query,
                start_time=start_time,
                end_time=end_time,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics', 'context_annotations']
            ).flatten(limit=500)  # Limit to 500 tweets
            
            tweet_texts = []
            for tweet in tweets:
                if tweet.text:
                    # Clean tweet text
                    cleaned_text = self._clean_text(tweet.text)
                    if len(cleaned_text) > 10:  # Only include meaningful tweets
                        tweet_texts.append(cleaned_text)
            
            if not tweet_texts:
                return {'sentiment': 0.0, 'mentions': 0}
            
            # Analyze sentiment using multiple methods
            sentiments = []
            
            for text in tweet_texts:
                # VADER sentiment
                vader_scores = self.vader_analyzer.polarity_scores(text)
                sentiments.append(vader_scores['compound'])
                
                # TextBlob sentiment
                try:
                    blob_sentiment = TextBlob(text).sentiment.polarity
                    sentiments.append(blob_sentiment)
                except:
                    pass
                
                # FinBERT sentiment (if available)
                if self.finbert_analyzer:
                    try:
                        finbert_result = self.finbert_analyzer(text)[0]
                        finbert_score = self._convert_finbert_sentiment(finbert_result)
                        sentiments.append(finbert_score)
                    except:
                        pass
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            return {
                'sentiment': avg_sentiment,
                'mentions': len(tweet_texts)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment for {symbol}: {str(e)}")
            return {'sentiment': 0.0, 'mentions': 0}
    
    async def _analyze_news_sentiment(self, symbol: str, keywords: List[str]) -> Dict:
        """Analyze news sentiment for a symbol"""
        try:
            if not self.news_api_key:
                return {'sentiment': 0.0, 'mentions': 0}
            
            logger.debug(f"Fetching news data for {symbol}")
            
            async with aiohttp.ClientSession() as session:
                # Search for news articles
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f"{symbol} OR {keywords[0] if keywords else symbol}",
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'from': (datetime.now() - timedelta(days=1)).isoformat(),
                    'to': datetime.now().isoformat(),
                    'apiKey': self.news_api_key,
                    'pageSize': 100
                }
                
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"News API returned status {response.status}")
                        return {'sentiment': 0.0, 'mentions': 0}
                    
                    data = await response.json()
                    articles = data.get('articles', [])
                
                if not articles:
                    return {'sentiment': 0.0, 'mentions': 0}
                
                # Analyze sentiment of headlines and descriptions
                sentiments = []
                
                for article in articles:
                    texts_to_analyze = []
                    
                    if article.get('title'):
                        texts_to_analyze.append(article['title'])
                    if article.get('description'):
                        texts_to_analyze.append(article['description'])
                    
                    for text in texts_to_analyze:
                        if text and len(text) > 10:
                            cleaned_text = self._clean_text(text)
                            
                            # VADER sentiment
                            vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
                            sentiments.append(vader_scores['compound'])
                            
                            # TextBlob sentiment
                            try:
                                blob_sentiment = TextBlob(cleaned_text).sentiment.polarity
                                sentiments.append(blob_sentiment)
                            except:
                                pass
                            
                            # FinBERT sentiment (if available)
                            if self.finbert_analyzer:
                                try:
                                    finbert_result = self.finbert_analyzer(cleaned_text)[0]
                                    finbert_score = self._convert_finbert_sentiment(finbert_result)
                                    sentiments.append(finbert_score)
                                except:
                                    pass
                
                # Calculate average sentiment
                avg_sentiment = np.mean(sentiments) if sentiments else 0.0
                
                return {
                    'sentiment': avg_sentiment,
                    'mentions': len(articles)
                }
                
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {str(e)}")
            return {'sentiment': 0.0, 'mentions': 0}
    
    async def _analyze_reddit_sentiment(self, symbol: str, keywords: List[str]) -> Dict:
        """Analyze Reddit sentiment for a symbol"""
        try:
            if not self.reddit_client_id or not self.reddit_client_secret:
                return {'sentiment': 0.0, 'mentions': 0}
            
            logger.debug(f"Fetching Reddit data for {symbol}")
            
            # Reddit API authentication
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                'grant_type': 'client_credentials'
            }
            
            async with aiohttp.ClientSession() as session:
                # Get access token
                auth = aiohttp.BasicAuth(self.reddit_client_id, self.reddit_client_secret)
                headers = {'User-Agent': os.getenv('REDDIT_USER_AGENT', 'TradingBot/1.0')}
                
                async with session.post(auth_url, data=auth_data, auth=auth, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"Reddit auth failed with status {response.status}")
                        return {'sentiment': 0.0, 'mentions': 0}
                    
                    token_data = await response.json()
                    access_token = token_data.get('access_token')
                
                if not access_token:
                    return {'sentiment': 0.0, 'mentions': 0}
                
                # Search Reddit posts
                headers['Authorization'] = f'bearer {access_token}'
                
                # Search in relevant subreddits
                subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting', 'wallstreetbets']
                all_posts = []
                
                for subreddit in subreddits:
                    search_url = f"https://oauth.reddit.com/r/{subreddit}/search"
                    params = {
                        'q': f"{symbol} OR {keywords[0] if keywords else symbol}",
                        'sort': 'new',
                        'limit': 25,
                        'restrict_sr': True,
                        't': 'day'  # Last 24 hours
                    }
                    
                    try:
                        async with session.get(search_url, params=params, headers=headers) as response:
                            if response.status == 200:
                                data = await response.json()
                                posts = data.get('data', {}).get('children', [])
                                all_posts.extend(posts)
                    except:
                        continue
                
                if not all_posts:
                    return {'sentiment': 0.0, 'mentions': 0}
                
                # Analyze sentiment
                sentiments = []
                
                for post in all_posts:
                    post_data = post.get('data', {})
                    texts_to_analyze = []
                    
                    if post_data.get('title'):
                        texts_to_analyze.append(post_data['title'])
                    if post_data.get('selftext') and len(post_data['selftext']) < 500:
                        texts_to_analyze.append(post_data['selftext'])
                    
                    for text in texts_to_analyze:
                        if text and len(text) > 10:
                            cleaned_text = self._clean_text(text)
                            
                            # VADER sentiment
                            vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
                            sentiments.append(vader_scores['compound'])
                            
                            # TextBlob sentiment
                            try:
                                blob_sentiment = TextBlob(cleaned_text).sentiment.polarity
                                sentiments.append(blob_sentiment)
                            except:
                                pass
                
                # Calculate average sentiment
                avg_sentiment = np.mean(sentiments) if sentiments else 0.0
                
                return {
                    'sentiment': avg_sentiment,
                    'mentions': len(all_posts)
                }
                
        except Exception as e:
            logger.error(f"Error analyzing Reddit sentiment for {symbol}: {str(e)}")
            return {'sentiment': 0.0, 'mentions': 0}
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (keep the word part)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def _convert_finbert_sentiment(self, finbert_result: Dict) -> float:
        """Convert FinBERT output to sentiment score (-1 to 1)"""
        label = finbert_result.get('label', 'neutral').lower()
        score = finbert_result.get('score', 0.0)
        
        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        else:  # neutral
            return 0.0
    
    def _calculate_composite_sentiment(self, sentiment_data: SentimentData) -> float:
        """Calculate weighted composite sentiment score"""
        try:
            sentiments = []
            weights = []
            
            # Twitter sentiment (weight based on mentions)
            if sentiment_data.twitter_sentiment is not None and sentiment_data.twitter_mentions > 0:
                sentiments.append(sentiment_data.twitter_sentiment)
                # Weight based on number of mentions (max weight = 0.5)
                twitter_weight = min(0.5, sentiment_data.twitter_mentions / 100)
                weights.append(twitter_weight)
            
            # News sentiment (usually more reliable, higher weight)
            if sentiment_data.news_sentiment is not None and sentiment_data.news_mentions > 0:
                sentiments.append(sentiment_data.news_sentiment)
                # News gets higher base weight
                news_weight = min(0.6, 0.4 + (sentiment_data.news_mentions / 50))
                weights.append(news_weight)
            
            # Reddit sentiment
            if sentiment_data.reddit_sentiment is not None and sentiment_data.reddit_mentions > 0:
                sentiments.append(sentiment_data.reddit_sentiment)
                reddit_weight = min(0.3, sentiment_data.reddit_mentions / 50)
                weights.append(reddit_weight)
            
            if not sentiments:
                return 0.0
            
            # Calculate weighted average
            if len(sentiments) == 1:
                return sentiments[0]
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                composite = sum(s * w for s, w in zip(sentiments, weights))
            else:
                composite = np.mean(sentiments)
            
            # Clamp to [-1, 1] range
            return max(-1.0, min(1.0, composite))
            
        except Exception as e:
            logger.error(f"Error calculating composite sentiment: {str(e)}")
            return 0.0
    
    def _calculate_sentiment_strength(self, sentiment_data: SentimentData) -> float:
        """Calculate sentiment strength/confidence (0-1)"""
        try:
            # Base strength on total mentions and sentiment magnitude
            total_mentions = (
                (sentiment_data.twitter_mentions or 0) +
                (sentiment_data.news_mentions or 0) +
                (sentiment_data.reddit_mentions or 0)
            )
            
            if total_mentions == 0:
                return 0.0
            
            # Mention-based strength (more mentions = higher confidence)
            mention_strength = min(1.0, total_mentions / 100)
            
            # Sentiment magnitude (stronger sentiments = higher confidence)
            sentiment_magnitude = abs(sentiment_data.composite_sentiment or 0.0)
            
            # Agreement between sources (if multiple sources agree, higher confidence)
            sentiments = []
            if sentiment_data.twitter_sentiment is not None:
                sentiments.append(sentiment_data.twitter_sentiment)
            if sentiment_data.news_sentiment is not None:
                sentiments.append(sentiment_data.news_sentiment)
            if sentiment_data.reddit_sentiment is not None:
                sentiments.append(sentiment_data.reddit_sentiment)
            
            agreement_strength = 1.0
            if len(sentiments) > 1:
                # Calculate standard deviation (lower = more agreement)
                std_dev = np.std(sentiments)
                agreement_strength = max(0.5, 1.0 - std_dev)
            
            # Combine factors
            overall_strength = (
                mention_strength * 0.4 +
                sentiment_magnitude * 0.3 +
                agreement_strength * 0.3
            )
            
            return min(1.0, overall_strength)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment strength: {str(e)}")
            return 0.0
    
    async def _create_empty_twitter_sentiment(self) -> Dict:
        """Create empty Twitter sentiment data"""
        return {'sentiment': 0.0, 'mentions': 0}
    
    async def _create_empty_news_sentiment(self) -> Dict:
        """Create empty news sentiment data"""
        return {'sentiment': 0.0, 'mentions': 0}
    
    async def _create_empty_reddit_sentiment(self) -> Dict:
        """Create empty Reddit sentiment data"""
        return {'sentiment': 0.0, 'mentions': 0}
    
    def generate_sentiment_score(self, sentiment_data: SentimentData) -> float:
        """
        Generate a trading score (0-100) based on sentiment data
        
        Args:
            sentiment_data: Sentiment data for a symbol
            
        Returns:
            Sentiment score from 0-100
        """
        try:
            if not sentiment_data.composite_sentiment:
                return 50  # Neutral score
            
            # Convert sentiment (-1 to 1) to score (0 to 100)
            base_score = ((sentiment_data.composite_sentiment + 1) / 2) * 100
            
            # Adjust based on sentiment strength (confidence)
            strength_factor = sentiment_data.sentiment_strength or 0.5
            
            # If low confidence, pull towards neutral (50)
            adjusted_score = 50 + (base_score - 50) * strength_factor
            
            # Ensure score is in valid range
            return max(0.0, min(100.0, adjusted_score))
            
        except Exception as e:
            logger.error(f"Error generating sentiment score: {str(e)}")
            return 50.0
    
    def get_sentiment_reasons(self, sentiment_data: SentimentData) -> List[str]:
        """Generate reasons for the sentiment score"""
        reasons = []
        
        try:
            composite = sentiment_data.composite_sentiment or 0.0
            strength = sentiment_data.sentiment_strength or 0.0
            
            # Overall sentiment
            if composite > 0.3:
                reasons.append("Positive social media sentiment")
            elif composite < -0.3:
                reasons.append("Negative social media sentiment")
            else:
                reasons.append("Neutral social media sentiment")
            
            # Mention volume
            total_mentions = (
                (sentiment_data.twitter_mentions or 0) +
                (sentiment_data.news_mentions or 0) +
                (sentiment_data.reddit_mentions or 0)
            )
            
            if total_mentions > 50:
                reasons.append("High social media activity")
            elif total_mentions > 10:
                reasons.append("Moderate social media activity")
            else:
                reasons.append("Low social media activity")
            
            # Source-specific insights
            if sentiment_data.news_mentions and sentiment_data.news_mentions > 5:
                if sentiment_data.news_sentiment and sentiment_data.news_sentiment > 0.2:
                    reasons.append("Positive news coverage")
                elif sentiment_data.news_sentiment and sentiment_data.news_sentiment < -0.2:
                    reasons.append("Negative news coverage")
            
            if sentiment_data.twitter_mentions and sentiment_data.twitter_mentions > 20:
                reasons.append("Active Twitter discussion")
            
            # Confidence level
            if strength > 0.7:
                reasons.append("High sentiment confidence")
            elif strength < 0.3:
                reasons.append("Low sentiment confidence")
                
        except Exception as e:
            logger.error(f"Error generating sentiment reasons: {str(e)}")
            reasons.append("Sentiment analysis completed")
        
        return reasons[:4]  # Limit to 4 reasons

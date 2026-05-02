"""
Sentiment analysis for stock prediction from news and social media.
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import newspaper
from newspaper import Article
import tweepy
import logging
from dataclasses import dataclass
import re

logger = logging.getLogger("stockvision.sentiment")

@dataclass
class SentimentScore:
    """Data class for sentiment analysis results."""
    ticker: str
    source: str  # 'news', 'twitter', 'reddit'
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    text: str
    timestamp: datetime
    url: Optional[str] = None

class NewsSentimentAnalyzer:
    """Analyze sentiment from financial news sources."""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.news_sources = [
            "https://finance.yahoo.com",
            "https://www.bloomberg.com",
            "https://www.reuters.com",
            "https://www.cnbc.com"
        ]
    
    async def search_news(self, ticker: str, days_back: int = 7) -> List[str]:
        """Search for news articles related to ticker."""
        # This is a simplified implementation
        # In production, you'd use news APIs like NewsAPI, Alpha Vantage News, etc.
        
        search_urls = []
        for source in self.news_sources:
            # Mock search results - replace with actual API calls
            search_urls.extend([
                f"{source}/search?q={ticker}&fr=srchsrp_top",
                f"{source}/quote/{ticker}/news"
            ])
        
        return search_urls[:10]  # Limit to 10 articles
    
    async def extract_article_text(self, url: str) -> Optional[str]:
        """Extract text from news article URL."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.error(f"Error extracting article from {url}: {e}")
            return None
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment of text using VADER and TextBlob."""
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        vader_sentiment = vader_scores['compound']
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        
        # Combine scores (weighted average)
        combined_sentiment = 0.6 * vader_sentiment + 0.4 * textblob_sentiment
        
        # Confidence based on VADER's compound score magnitude
        confidence = abs(vader_sentiment)
        
        return combined_sentiment, confidence
    
    async def analyze_news_sentiment(self, ticker: str) -> List[SentimentScore]:
        """Analyze sentiment from news articles for a ticker."""
        sentiment_scores = []
        
        try:
            # Search for news
            news_urls = await self.search_news(ticker)
            
            # Extract and analyze each article
            for url in news_urls:
                text = await self.extract_article_text(url)
                if text and len(text) > 100:  # Ensure meaningful content
                    sentiment, confidence = self.analyze_sentiment(text)
                    
                    sentiment_scores.append(SentimentScore(
                        ticker=ticker,
                        source="news",
                        sentiment_score=sentiment,
                        confidence=confidence,
                        text=text[:500],  # Store first 500 chars
                        timestamp=datetime.utcnow(),
                        url=url
                    ))
                    
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {ticker}: {e}")
        
        return sentiment_scores

class TwitterSentimentAnalyzer:
    """Analyze sentiment from Twitter."""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize Twitter client
        try:
            auth = tweepy.OAuthHandler(api_key, api_secret)
            auth.set_access_token(access_token, access_token_secret)
            self.client = tweepy.API(auth, wait_on_rate_limit=True)
        except Exception as e:
            logger.error(f"Error initializing Twitter client: {e}")
            self.client = None
    
    def search_tweets(self, ticker: str, count: int = 100) -> List[str]:
        """Search for tweets about ticker."""
        if not self.client:
            return []
        
        try:
            # Search for tweets with ticker symbol
            query = f"${ticker} OR {ticker} stock"
            tweets = tweepy.Cursor(
                self.client.search_tweets,
                q=query,
                lang="en",
                result_type="recent",
                tweet_mode="extended"
            ).items(count)
            
            tweet_texts = []
            for tweet in tweets:
                # Clean tweet text
                text = self.clean_tweet_text(tweet.full_text)
                if text:
                    tweet_texts.append(text)
            
            return tweet_texts
            
        except Exception as e:
            logger.error(f"Error searching tweets for {ticker}: {e}")
            return []
    
    def clean_tweet_text(self, text: str) -> str:
        """Clean tweet text by removing URLs, mentions, etc."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        text = re.sub(r'\@\w+|\#', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def analyze_tweet_sentiment(self, tweet_text: str) -> Tuple[float, float]:
        """Analyze sentiment of a single tweet."""
        vader_scores = self.vader.polarity_scores(tweet_text)
        sentiment = vader_scores['compound']
        confidence = abs(sentiment)
        
        return sentiment, confidence
    
    async def analyze_twitter_sentiment(self, ticker: str) -> List[SentimentScore]:
        """Analyze sentiment from Twitter for a ticker."""
        sentiment_scores = []
        
        try:
            tweets = self.search_tweets(ticker)
            
            for tweet_text in tweets:
                sentiment, confidence = self.analyze_tweet_sentiment(tweet_text)
                
                sentiment_scores.append(SentimentScore(
                    ticker=ticker,
                    source="twitter",
                    sentiment_score=sentiment,
                    confidence=confidence,
                    text=tweet_text[:200],  # Store first 200 chars
                    timestamp=datetime.utcnow()
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment for {ticker}: {e}")
        
        return sentiment_scores

class SentimentAggregator:
    """Aggregate sentiment from multiple sources."""
    
    def __init__(self, twitter_config: Optional[Dict] = None):
        self.news_analyzer = NewsSentimentAnalyzer()
        self.twitter_analyzer = None
        
        if twitter_config:
            self.twitter_analyzer = TwitterSentimentAnalyzer(
                twitter_config.get('api_key'),
                twitter_config.get('api_secret'),
                twitter_config.get('access_token'),
                twitter_config.get('access_token_secret')
            )
    
    async def get_comprehensive_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis from all sources."""
        all_sentiments = []
        
        # Get news sentiment
        news_sentiments = await self.news_analyzer.analyze_news_sentiment(ticker)
        all_sentiments.extend(news_sentiments)
        
        # Get Twitter sentiment if available
        if self.twitter_analyzer:
            twitter_sentiments = await self.twitter_analyzer.analyze_twitter_sentiment(ticker)
            all_sentiments.extend(twitter_sentiments)
        
        if not all_sentiments:
            return {
                "ticker": ticker,
                "overall_sentiment": 0.0,
                "confidence": 0.0,
                "source_breakdown": {},
                "total_articles": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Calculate overall sentiment
        weighted_sentiments = []
        for sentiment in all_sentiments:
            weight = sentiment.confidence
            weighted_sentiments.append(sentiment.sentiment_score * weight)
        
        overall_sentiment = np.mean(weighted_sentiments) if weighted_sentiments else 0.0
        avg_confidence = np.mean([s.confidence for s in all_sentiments])
        
        # Breakdown by source
        source_breakdown = {}
        for source in ["news", "twitter"]:
            source_sentiments = [s for s in all_sentiments if s.source == source]
            if source_sentiments:
                source_avg = np.mean([s.sentiment_score for s in source_sentiments])
                source_count = len(source_sentiments)
                source_breakdown[source] = {
                    "sentiment": float(source_avg),
                    "count": source_count
                }
        
        return {
            "ticker": ticker,
            "overall_sentiment": float(overall_sentiment),
            "confidence": float(avg_confidence),
            "source_breakdown": source_breakdown,
            "total_articles": len(all_sentiments),
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment_label": self._get_sentiment_label(overall_sentiment)
        }
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label."""
        if sentiment_score > 0.1:
            return "BULLISH"
        elif sentiment_score < -0.1:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def get_sentiment_trend(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """Get sentiment trend over time (mock implementation)."""
        # In production, this would query historical sentiment data
        # For now, return mock trend data
        
        dates = pd.date_range(end=datetime.utcnow(), periods=days, freq='D')
        sentiment_trend = []
        
        for date in dates:
            # Mock sentiment with some randomness
            base_sentiment = np.random.normal(0.1, 0.2)  # Slightly bullish bias
            sentiment_trend.append({
                "date": date.strftime("%Y-%m-%d"),
                "sentiment": float(base_sentiment),
                "confidence": float(np.random.uniform(0.5, 0.9))
            })
        
        return {
            "ticker": ticker,
            "trend": sentiment_trend,
            "period_days": days
        }

# Sentiment-adjusted prediction integration
class SentimentAdjustedPredictor:
    """Adjust stock predictions based on sentiment analysis."""
    
    def __init__(self, sentiment_aggregator: SentimentAggregator):
        self.sentiment_aggregator = sentiment_aggregator
    
    async def adjust_prediction(self, 
                               original_prediction: float, 
                               ticker: str, 
                               sentiment_weight: float = 0.1) -> Dict[str, Any]:
        """Adjust stock prediction based on sentiment."""
        # Get sentiment analysis
        sentiment_data = await self.sentiment_aggregator.get_comprehensive_sentiment(ticker)
        
        # Calculate sentiment adjustment
        sentiment_score = sentiment_data["overall_sentiment"]
        confidence = sentiment_data["confidence"]
        
        # Apply sentiment adjustment
        adjustment_factor = sentiment_weight * sentiment_score * confidence
        adjusted_prediction = original_prediction * (1 + adjustment_factor)
        
        return {
            "original_prediction": original_prediction,
            "adjusted_prediction": adjusted_prediction,
            "sentiment_data": sentiment_data,
            "adjustment_factor": adjustment_factor,
            "sentiment_weight": sentiment_weight
        }

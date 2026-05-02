"""
Financial news aggregation from multiple sources.
"""
import asyncio
import aiohttp
import requests
import feedparser
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re
from newspaper import Article
import time

logger = logging.getLogger("stockvision.news_aggregator")

@dataclass
class NewsArticle:
    """Data class for news articles."""
    title: str
    url: str
    source: str
    published_date: datetime
    summary: str
    content: str
    tickers: List[str]
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    author: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class ReutersNewsExtractor:
    """Extract financial news from Reuters."""
    
    def __init__(self):
        self.base_url = "https://www.reuters.com"
        self.markets_url = "https://www.reuters.com/markets"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_market_news(self, limit: int = 20) -> List[NewsArticle]:
        """Get latest market news from Reuters."""
        try:
            # Get business news feed
            feed_url = "https://www.reuters.com/rssfeed/marketsNews"
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries[:limit]:
                try:
                    # Extract full article content
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    
                    # Extract tickers from content
                    tickers = self._extract_tickers(article.text)
                    
                    news_article = NewsArticle(
                        title=entry.title,
                        url=entry.link,
                        source="Reuters",
                        published_date=datetime(*entry.published_parsed[:6]) if entry.published_parsed else datetime.utcnow(),
                        summary=article.summary or entry.summary,
                        content=article.text,
                        tickers=tickers,
                        author=getattr(article, 'authors', [None])[0]
                    )
                    articles.append(news_article)
                    
                except Exception as e:
                    logger.warning(f"Error processing Reuters article {entry.title}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Reuters news: {e}")
            return []
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text."""
        # Pattern for stock tickers (e.g., AAPL, GOOGL, MSFT)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter out common words that match ticker pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'HAD', 'HAS', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        tickers = [ticker for ticker in potential_tickers if ticker not in common_words]
        return list(set(tickers))  # Remove duplicates

class BloombergNewsExtractor:
    """Extract financial news from Bloomberg."""
    
    def __init__(self):
        self.base_url = "https://www.bloomberg.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_market_news(self, limit: int = 20) -> List[NewsArticle]:
        """Get latest market news from Bloomberg."""
        try:
            # Bloomberg RSS feed (if available)
            feed_url = "https://feeds.bloomberg.com/markets/news.rss"
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries[:limit]:
                try:
                    # Extract content from link
                    response = self.session.get(entry.link, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract article content
                    content_elements = soup.select('.body-content p')
                    content = ' '.join([p.get_text() for p in content_elements])
                    
                    # Extract tickers
                    tickers = self._extract_tickers(entry.title + ' ' + content)
                    
                    news_article = NewsArticle(
                        title=entry.title,
                        url=entry.link,
                        source="Bloomberg",
                        published_date=datetime(*entry.published_parsed[:6]) if entry.published_parsed else datetime.utcnow(),
                        summary=entry.summary,
                        content=content,
                        tickers=tickers
                    )
                    articles.append(news_article)
                    
                except Exception as e:
                    logger.warning(f"Error processing Bloomberg article {entry.title}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Bloomberg news: {e}")
            return []
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text."""
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'HAD', 'HAS', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        tickers = [ticker for ticker in potential_tickers if ticker not in common_words]
        return list(set(tickers))

class YahooNewsExtractor:
    """Extract financial news from Yahoo Finance."""
    
    def __init__(self):
        self.base_url = "https://finance.yahoo.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_market_news(self, limit: int = 20) -> List[NewsArticle]:
        """Get latest market news from Yahoo Finance."""
        try:
            # Yahoo Finance RSS feed
            feed_url = "https://finance.yahoo.com/rss/rss2.aspx"
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries[:limit]:
                try:
                    # Extract full article content
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    
                    # Extract tickers
                    tickers = self._extract_tickers(article.text)
                    
                    news_article = NewsArticle(
                        title=entry.title,
                        url=entry.link,
                        source="Yahoo Finance",
                        published_date=datetime(*entry.published_parsed[:6]) if entry.published_parsed else datetime.utcnow(),
                        summary=article.summary or entry.summary,
                        content=article.text,
                        tickers=tickers,
                        author=getattr(article, 'authors', [None])[0]
                    )
                    articles.append(news_article)
                    
                except Exception as e:
                    logger.warning(f"Error processing Yahoo Finance article {entry.title}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {e}")
            return []
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text."""
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'HAD', 'HAS', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        tickers = [ticker for ticker in potential_tickers if ticker not in common_words]
        return list(set(tickers))

class SeekingAlphaExtractor:
    """Extract financial news from Seeking Alpha."""
    
    def __init__(self):
        self.base_url = "https://seekingalpha.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_market_news(self, limit: int = 20) -> List[NewsArticle]:
        """Get latest market news from Seeking Alpha."""
        try:
            # Seeking Alpha RSS feed
            feed_url = "https://seekingalpha.com/feed.xml"
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries[:limit]:
                try:
                    # Extract content from link
                    response = self.session.get(entry.link, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract article content
                    content_elements = soup.select('.paywall-content p')
                    content = ' '.join([p.get_text() for p in content_elements])
                    
                    # Extract tickers
                    tickers = self._extract_tickers(entry.title + ' ' + content)
                    
                    news_article = NewsArticle(
                        title=entry.title,
                        url=entry.link,
                        source="Seeking Alpha",
                        published_date=datetime(*entry.published_parsed[:6]) if entry.published_parsed else datetime.utcnow(),
                        summary=entry.summary,
                        content=content,
                        tickers=tickers
                    )
                    articles.append(news_article)
                    
                except Exception as e:
                    logger.warning(f"Error processing Seeking Alpha article {entry.title}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Seeking Alpha news: {e}")
            return []
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text."""
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'HAD', 'HAS', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        tickers = [ticker for ticker in potential_tickers if ticker not in common_words]
        return list(set(tickers))

class NewsAggregator:
    """Aggregate news from multiple financial sources."""
    
    def __init__(self):
        self.extractors = {
            'reuters': ReutersNewsExtractor(),
            'yahoo': YahooNewsExtractor(),
            'bloomberg': BloombergNewsExtractor(),
            'seeking_alpha': SeekingAlphaExtractor()
        }
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_all_news(self, limit_per_source: int = 10, sources: List[str] = None) -> List[NewsArticle]:
        """Get news from all sources."""
        sources = sources or list(self.extractors.keys())
        all_articles = []
        
        for source in sources:
            if source in self.extractors:
                try:
                    articles = self.extractors[source].get_market_news(limit_per_source)
                    all_articles.extend(articles)
                    logger.info(f"Fetched {len(articles)} articles from {source}")
                except Exception as e:
                    logger.error(f"Error fetching news from {source}: {e}")
        
        # Sort by published date
        all_articles.sort(key=lambda x: x.published_date, reverse=True)
        
        return all_articles
    
    def get_news_by_ticker(self, ticker: str, hours_back: int = 24) -> List[NewsArticle]:
        """Get news related to a specific ticker."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        # Get recent news
        recent_news = self.get_all_news(limit_per_source=20)
        
        # Filter by ticker
        ticker_news = []
        for article in recent_news:
            if article.published_date > cutoff_time and ticker.upper() in [t.upper() for t in article.tickers]:
                ticker_news.append(article)
        
        return ticker_news
    
    def get_trending_stocks(self, hours_back: int = 24) -> Dict[str, int]:
        """Get trending stocks based on news mentions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_news = self.get_all_news(limit_per_source=30)
        
        # Count ticker mentions
        ticker_counts = {}
        for article in recent_news:
            if article.published_date > cutoff_time:
                for ticker in article.tickers:
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        
        # Sort by mention count
        trending = dict(sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True))
        
        return trending
    
    def get_market_sentiment(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get overall market sentiment from news."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_news = self.get_all_news(limit_per_source=20)
        
        # Filter recent news
        recent_news = [article for article in recent_news if article.published_date > cutoff_time]
        
        if not recent_news:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "NEUTRAL",
                "article_count": 0,
                "top_tickers": {}
            }
        
        # Simple sentiment analysis (would be enhanced with proper NLP)
        positive_words = ['up', 'rise', 'gain', 'bull', 'growth', 'strong', 'positive', 'higher']
        negative_words = ['down', 'fall', 'loss', 'bear', 'decline', 'weak', 'negative', 'lower']
        
        total_sentiment = 0
        ticker_sentiments = {}
        
        for article in recent_news:
            # Calculate sentiment for this article
            content = (article.title + ' ' + article.summary).lower()
            positive_count = sum(1 for word in positive_words if word in content)
            negative_count = sum(1 for word in negative_words if word in content)
            
            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment = 0.0
            
            total_sentiment += sentiment
            
            # Add to ticker sentiments
            for ticker in article.tickers:
                if ticker not in ticker_sentiments:
                    ticker_sentiments[ticker] = []
                ticker_sentiments[ticker].append(sentiment)
        
        # Calculate average sentiment
        avg_sentiment = total_sentiment / len(recent_news) if recent_news else 0.0
        
        # Determine sentiment label
        if avg_sentiment > 0.1:
            sentiment_label = "BULLISH"
        elif avg_sentiment < -0.1:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"
        
        # Calculate average sentiment per ticker
        top_tickers = {}
        for ticker, sentiments in ticker_sentiments.items():
            if len(sentiments) >= 2:  # At least 2 mentions
                avg_ticker_sentiment = sum(sentiments) / len(sentiments)
                top_tickers[ticker] = {
                    "sentiment": avg_ticker_sentiment,
                    "mentions": len(sentiments)
                }
        
        # Sort top tickers by sentiment
        top_tickers = dict(sorted(top_tickers.items(), 
                                key=lambda x: x[1]["sentiment"], 
                                reverse=True)[:10])
        
        return {
            "sentiment_score": avg_sentiment,
            "sentiment_label": sentiment_label,
            "article_count": len(recent_news),
            "top_tickers": top_tickers,
            "timestamp": datetime.utcnow().isoformat()
        }

# Global news aggregator instance
news_aggregator = NewsAggregator()

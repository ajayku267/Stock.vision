"""
Real-time financial data extraction from multiple sources.
"""
import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
import yfinance as yf
from bs4 import BeautifulSoup
import time

logger = logging.getLogger("stockvision.data_extractors")

@dataclass
class StockData:
    """Data class for stock information."""
    ticker: str
    company_name: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    avg_volume: Optional[int] = None
    beta: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class GoogleFinanceExtractor:
    """Extract real-time data from Google Finance."""
    
    def __init__(self):
        self.base_url = "https://www.google.com/finance/quote"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_stock_data(self, ticker: str) -> Optional[StockData]:
        """Extract stock data from Google Finance."""
        try:
            url = f"{self.base_url}/{ticker}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract company name
            company_name = self._extract_company_name(soup)
            
            # Extract current price and change
            price_data = self._extract_price_data(soup)
            
            # Extract key statistics
            stats = self._extract_key_statistics(soup)
            
            return StockData(
                ticker=ticker.upper(),
                company_name=company_name,
                current_price=price_data['price'],
                change=price_data['change'],
                change_percent=price_data['change_percent'],
                volume=stats.get('volume', 0),
                market_cap=stats.get('market_cap'),
                pe_ratio=stats.get('pe_ratio'),
                dividend_yield=stats.get('dividend_yield'),
                high_52w=stats.get('high_52w'),
                low_52w=stats.get('low_52w'),
                avg_volume=stats.get('avg_volume'),
                beta=stats.get('beta'),
                eps=stats.get('eps'),
                revenue=stats.get('revenue')
            )
            
        except Exception as e:
            logger.error(f"Error extracting data from Google Finance for {ticker}: {e}")
            return None
    
    def _extract_company_name(self, soup: BeautifulSoup) -> str:
        """Extract company name from Google Finance page."""
        try:
            # Try multiple selectors for company name
            selectors = [
                'div[data-attr-id="title"] h1',
                'h1[data-attr-id="title"]',
                '.zzDege',
                'h1'
            ]
            
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    return element.get_text().strip()
            
            return "Unknown Company"
        except:
            return "Unknown Company"
    
    def _extract_price_data(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Extract price and change data."""
        try:
            # Current price
            price_selectors = [
                '.YMlKec.fxKbKc',
                'div[data-attr-id="price"] span',
                '.AHmHk .fxKbKc'
            ]
            
            price = 0.0
            for selector in price_selectors:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text().replace(',', '').replace('$', '')
                    price = float(price_text)
                    break
            
            # Change and change percentage
            change_selectors = [
                '.JwB6zf',
                'div[data-attr-id="price-change"] span',
                '.P2Luy'
            ]
            
            change = 0.0
            change_percent = 0.0
            
            for selector in change_selectors:
                element = soup.select_one(selector)
                if element:
                    change_text = element.get_text()
                    # Parse change text like "+5.25 (+2.1%)"
                    if '+' in change_text or '-' in change_text:
                        parts = change_text.split()
                        if len(parts) >= 2:
                            try:
                                change = float(parts[0].replace('+', '').replace(',', ''))
                                change_percent = float(parts[1].replace('(', '').replace(')', '').replace('%', ''))
                            except:
                                pass
                    break
            
            return {
                'price': price,
                'change': change,
                'change_percent': change_percent
            }
            
        except Exception as e:
            logger.error(f"Error extracting price data: {e}")
            return {'price': 0.0, 'change': 0.0, 'change_percent': 0.0}
    
    def _extract_key_statistics(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract key statistics from the page."""
        stats = {}
        
        try:
            # Look for data in the key statistics section
            data_rows = soup.select('div[data-attr-id="key-statistics"] tr')
            
            for row in data_rows:
                cells = row.select('td')
                if len(cells) >= 2:
                    label = cells[0].get_text().strip().lower()
                    value = cells[1].get_text().strip()
                    
                    # Parse different statistics
                    if 'volume' in label and 'avg' not in label:
                        stats['volume'] = self._parse_number(value)
                    elif 'avg volume' in label:
                        stats['avg_volume'] = self._parse_number(value)
                    elif 'market cap' in label:
                        stats['market_cap'] = self._parse_number(value)
                    elif 'pe ratio' in label:
                        stats['pe_ratio'] = self._parse_number(value)
                    elif 'dividend yield' in label:
                        stats['dividend_yield'] = self._parse_number(value)
                    elif '52 week high' in label:
                        stats['high_52w'] = self._parse_number(value)
                    elif '52 week low' in label:
                        stats['low_52w'] = self._parse_number(value)
                    elif 'beta' in label:
                        stats['beta'] = self._parse_number(value)
                    elif 'eps' in label:
                        stats['eps'] = self._parse_number(value)
                    elif 'revenue' in label:
                        stats['revenue'] = self._parse_number(value)
            
        except Exception as e:
            logger.error(f"Error extracting key statistics: {e}")
        
        return stats
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Parse number from text (handles K, M, B, T suffixes)."""
        try:
            text = text.replace(',', '').replace('$', '').replace('%', '')
            
            if 'T' in text:
                return float(text.replace('T', '')) * 1_000_000_000_000
            elif 'B' in text:
                return float(text.replace('B', '')) * 1_000_000_000
            elif 'M' in text:
                return float(text.replace('M', '')) * 1_000_000
            elif 'K' in text:
                return float(text.replace('K', '')) * 1_000
            else:
                return float(text)
        except:
            return None

class YahooFinanceExtractor:
    """Enhanced Yahoo Finance data extractor."""
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_stock_data(self, ticker: str) -> Optional[StockData]:
        """Extract comprehensive stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get current data
            info = stock.info
            history = stock.history(period="1d")
            
            if history.empty:
                return None
            
            current_price = history['Close'].iloc[-1]
            previous_close = history['Open'].iloc[-1]
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
            
            return StockData(
                ticker=ticker.upper(),
                company_name=info.get('longName', 'Unknown Company'),
                current_price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(history['Volume'].iloc[-1]),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield'),
                high_52w=info.get('fiftyTwoWeekHigh'),
                low_52w=info.get('fiftyTwoWeekLow'),
                avg_volume=info.get('averageVolume'),
                beta=info.get('beta'),
                eps=info.get('trailingEps'),
                revenue=info.get('totalRevenue')
            )
            
        except Exception as e:
            logger.error(f"Error extracting data from Yahoo Finance for {ticker}: {e}")
            return None

class MarketWatchExtractor:
    """Extract data from MarketWatch."""
    
    def __init__(self):
        self.base_url = "https://www.marketwatch.com/investing/stock"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_data(self, ticker: str) -> Optional[StockData]:
        """Extract stock data from MarketWatch."""
        try:
            url = f"{self.base_url}/{ticker}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract price data
            price_element = soup.select_one('.intraday__price .value')
            current_price = float(price_element.get_text().replace(',', '').replace('$', '')) if price_element else 0.0
            
            # Extract change data
            change_element = soup.select_one('.intraday__change .value')
            if change_element:
                change_text = change_element.get_text()
                change_parts = change_text.split()
                change = float(change_parts[0].replace('+', '').replace(',', ''))
                change_percent = float(change_parts[1].replace('(', '').replace(')', '').replace('%', ''))
            else:
                change = 0.0
                change_percent = 0.0
            
            # Extract company name
            company_element = soup.select_one('.company__name')
            company_name = company_element.get_text().strip() if company_element else "Unknown Company"
            
            return StockData(
                ticker=ticker.upper(),
                company_name=company_name,
                current_price=current_price,
                change=change,
                change_percent=change_percent,
                volume=0,  # Would need additional parsing
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error extracting data from MarketWatch for {ticker}: {e}")
            return None

class FinnhubExtractor:
    """Extract data using Finnhub API (free tier available)."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "demo"  # Use demo key for testing
        self.base_url = "https://finnhub.io/api/v1"
    
    def get_stock_data(self, ticker: str) -> Optional[StockData]:
        """Extract stock data from Finnhub API."""
        try:
            # Get quote data
            quote_url = f"{self.base_url}/quote"
            quote_params = {'symbol': ticker, 'token': self.api_key}
            quote_response = requests.get(quote_url, params=quote_params, timeout=10)
            quote_response.raise_for_status()
            quote_data = quote_response.json()
            
            # Get profile data
            profile_url = f"{self.base_url}/stock/profile2"
            profile_params = {'symbol': ticker, 'token': self.api_key}
            profile_response = requests.get(profile_url, params=profile_params, timeout=10)
            profile_response.raise_for_status()
            profile_data = profile_response.json()
            
            # Get metrics data
            metrics_url = f"{self.base_url}/stock/metric"
            metrics_params = {'symbol': ticker, 'token': self.api_key, 'metric': 'all'}
            metrics_response = requests.get(metrics_url, params=metrics_params, timeout=10)
            metrics_data = metrics_response.json() if metrics_response.status_code == 200 else {}
            
            current_price = quote_data.get('c', 0)
            previous_close = quote_data.get('pc', current_price)
            change = quote_data.get('d', 0)
            change_percent = quote_data.get('dp', 0)
            
            return StockData(
                ticker=ticker.upper(),
                company_name=profile_data.get('name', 'Unknown Company'),
                current_price=current_price,
                change=change,
                change_percent=change_percent,
                volume=quote_data.get('vol', 0),
                market_cap=profile_data.get('marketCapitalization'),
                high_52w=metrics_data.get('metric', {}).get('52WeekHigh'),
                low_52w=metrics_data.get('metric', {}).get('52WeekLow'),
                beta=metrics_data.get('metric', {}).get('beta'),
                eps=metrics_data.get('metric', {}).get('epsBasicExclExtraTTM')
            )
            
        except Exception as e:
            logger.error(f"Error extracting data from Finnhub for {ticker}: {e}")
            return None

class MultiSourceDataExtractor:
    """Combine multiple data sources for comprehensive financial data."""
    
    def __init__(self, finnhub_api_key: str = None):
        self.extractors = {
            'yahoo': YahooFinanceExtractor(),
            'google': GoogleFinanceExtractor(),
            'marketwatch': MarketWatchExtractor(),
            'finnhub': FinnhubExtractor(finnhub_api_key) if finnhub_api_key else None
        }
        self.cache = {}
        self.cache_timeout = 60  # seconds
    
    def get_stock_data(self, ticker: str, preferred_sources: List[str] = None) -> Optional[StockData]:
        """Get stock data from multiple sources with fallback."""
        ticker = ticker.upper()
        
        # Check cache first
        cache_key = f"{ticker}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Define source priority
        sources = preferred_sources or ['yahoo', 'google', 'marketwatch', 'finnhub']
        
        for source in sources:
            if source in self.extractors and self.extractors[source]:
                try:
                    data = self.extractors[source].get_stock_data(ticker)
                    if data and data.current_price > 0:
                        # Cache the result
                        self.cache[cache_key] = data
                        logger.info(f"Successfully extracted data for {ticker} from {source}")
                        return data
                except Exception as e:
                    logger.warning(f"Failed to extract data for {ticker} from {source}: {e}")
                    continue
        
        logger.error(f"Failed to extract data for {ticker} from all sources")
        return None
    
    def get_multiple_stocks(self, tickers: List[str], preferred_sources: List[str] = None) -> Dict[str, Optional[StockData]]:
        """Get data for multiple stocks concurrently."""
        results = {}
        
        for ticker in tickers:
            results[ticker] = self.get_stock_data(ticker, preferred_sources)
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview data."""
        market_indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
        indices_data = self.get_multiple_stocks(market_indices)
        
        return {
            'indices': indices_data,
            'timestamp': datetime.utcnow().isoformat(),
            'market_status': self._get_market_status()
        }
    
    def _get_market_status(self) -> str:
        """Get current market status."""
        now = datetime.utcnow()
        # Simplified market hours check (NYSE/NASDAQ)
        eastern_time = now - timedelta(hours=5)  # UTC-5 for EST
        weekday = eastern_time.weekday()
        hour = eastern_time.hour
        
        if weekday >= 5:  # Weekend
            return "Closed (Weekend)"
        elif 9 <= hour < 16:  # 9:00 AM - 4:00 PM EST
            return "Open"
        else:
            return "Closed"

class RealTimeDataStream:
    """Real-time data streaming for multiple stocks."""
    
    def __init__(self, extractor: MultiSourceDataExtractor):
        self.extractor = extractor
        self.subscribers = {}
        self.running = False
    
    def subscribe(self, ticker: str, callback):
        """Subscribe to real-time updates for a ticker."""
        if ticker not in self.subscribers:
            self.subscribers[ticker] = []
        self.subscribers[ticker].append(callback)
    
    def unsubscribe(self, ticker: str, callback):
        """Unsubscribe from updates for a ticker."""
        if ticker in self.subscribers:
            self.subscribers[ticker].remove(callback)
    
    async def start_streaming(self, update_interval: int = 30):
        """Start real-time data streaming."""
        self.running = True
        logger.info("Starting real-time data streaming")
        
        while self.running:
            try:
                tickers = list(self.subscribers.keys())
                if tickers:
                    data = self.extractor.get_multiple_stocks(tickers)
                    
                    for ticker, stock_data in data.items():
                        if ticker in self.subscribers and stock_data:
                            for callback in self.subscribers[ticker]:
                                try:
                                    callback(stock_data)
                                except Exception as e:
                                    logger.error(f"Error in callback for {ticker}: {e}")
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(5)
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        self.running = False
        logger.info("Stopped real-time data streaming")

# Global extractor instance
data_extractor = MultiSourceDataExtractor()
real_time_stream = RealTimeDataStream(data_extractor)

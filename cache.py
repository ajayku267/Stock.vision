"""
Simple in-memory caching layer for API responses.
"""
import time
import hashlib
import json
from typing import Any, Optional, Dict
from datetime import datetime, timedelta


class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default TTL
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_data = {k: v for k, v in sorted(kwargs.items())}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, prefix: str, **kwargs) -> Optional[Any]:
        """Get cached value if exists and not expired."""
        key = self._generate_key(prefix, **kwargs)
        
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if time.time() > entry['expires_at']:
            del self._cache[key]
            return None
        
        return entry['value']
    
    def set(self, prefix: str, value: Any, ttl: Optional[int] = None, **kwargs) -> None:
        """Set cached value with TTL."""
        key = self._generate_key(prefix, **kwargs)
        ttl = ttl or self.default_ttl
        
        self._cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
    
    def clear(self, prefix: Optional[str] = None) -> None:
        """Clear cache entries."""
        if prefix:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{prefix}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        total_entries = len(self._cache)
        expired_entries = sum(
            1 for entry in self._cache.values()
            if current_time > entry['expires_at']
        )
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'active_entries': total_entries - expired_entries
        }


# Global cache instance
cache = SimpleCache(default_ttl=300)  # 5 minutes


def cached_prediction(ticker: str, start_date: str, end_date: str, 
                     years_to_predict: int, backtest_days: int, 
                     signal_threshold_pct: float) -> Optional[Dict[str, Any]]:
    """Get cached prediction if available."""
    return cache.get(
        'prediction',
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        years_to_predict=years_to_predict,
        backtest_days=backtest_days,
        signal_threshold_pct=signal_threshold_pct
    )


def cache_prediction(prediction_data: Dict[str, Any], ticker: str, start_date: str, 
                    end_date: str, years_to_predict: int, backtest_days: int, 
                    signal_threshold_pct: float, ttl: int = 300) -> None:
    """Cache prediction data."""
    cache.set(
        'prediction',
        prediction_data,
        ttl=ttl,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        years_to_predict=years_to_predict,
        backtest_days=backtest_days,
        signal_threshold_pct=signal_threshold_pct
    )


def cached_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[Any]:
    """Get cached stock data if available."""
    return cache.get(
        'stock_data',
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )


def cache_stock_data(data: Any, ticker: str, start_date: str, end_date: str, 
                    ttl: int = 600) -> None:  # 10 minutes for stock data
    """Cache stock data."""
    cache.set(
        'stock_data',
        data,
        ttl=ttl,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )

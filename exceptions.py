"""
Custom exception classes for StockVision application.
"""
from typing import Optional, Any


class StockVisionException(Exception):
    """Base exception class for StockVision."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class DataValidationError(StockVisionException):
    """Raised when input data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, "DATA_VALIDATION_ERROR", details)


class InsufficientDataError(StockVisionException):
    """Raised when there's not enough data to perform an operation."""
    
    def __init__(self, message: str, required_rows: Optional[int] = None, actual_rows: Optional[int] = None):
        details = {}
        if required_rows is not None:
            details["required_rows"] = required_rows
        if actual_rows is not None:
            details["actual_rows"] = actual_rows
        super().__init__(message, "INSUFFICIENT_DATA", details)


class DataSourceError(StockVisionException):
    """Raised when external data source fails."""
    
    def __init__(self, message: str, source: Optional[str] = None, ticker: Optional[str] = None):
        details = {}
        if source:
            details["source"] = source
        if ticker:
            details["ticker"] = ticker
        super().__init__(message, "DATA_SOURCE_ERROR", details)


class ModelTrainingError(StockVisionException):
    """Raised when model training fails."""
    
    def __init__(self, message: str, model_type: Optional[str] = None):
        details = {}
        if model_type:
            details["model_type"] = model_type
        super().__init__(message, "MODEL_TRAINING_ERROR", details)


class BacktestError(StockVisionException):
    """Raised when backtesting fails."""
    
    def __init__(self, message: str, test_days: Optional[int] = None):
        details = {}
        if test_days is not None:
            details["test_days"] = test_days
        super().__init__(message, "BACKTEST_ERROR", details)


class TaskNotFoundError(StockVisionException):
    """Raised when an async task is not found."""
    
    def __init__(self, task_id: str):
        super().__init__(f"Task {task_id} not found", "TASK_NOT_FOUND", {"task_id": task_id})


class TaskTimeoutError(StockVisionException):
    """Raised when an async task times out."""
    
    def __init__(self, task_id: str, timeout_seconds: int):
        super().__init__(
            f"Task {task_id} timed out after {timeout_seconds} seconds",
            "TASK_TIMEOUT",
            {"task_id": task_id, "timeout_seconds": timeout_seconds}
        )


class RateLimitExceededError(StockVisionException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window_seconds: int, retry_after: int):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            "RATE_LIMIT_EXCEEDED",
            {
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after
            }
        )

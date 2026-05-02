"""
Celery tasks for distributed background processing.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from celery import current_task
import yfinance as yf
import pandas as pd

from celery_app import celery_app
from ml_models import EnsemblePredictor
from sentiment_analyzer import SentimentAggregator
from alert_system import AlertManager
from services import load_data, add_technical_indicators

logger = logging.getLogger("stockvision.celery_tasks")

# ML Prediction Tasks
@celery_app.task(bind=True, name="celery_tasks.ml_prediction")
def ml_prediction_task(self, ticker: str, start_date: str, end_date: str, 
                      years_to_predict: int = 1, backtest_days: int = 30,
                      signal_threshold_pct: float = 2.0) -> Dict[str, Any]:
    """Run ML ensemble prediction task."""
    try:
        # Update task progress
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Loading data"})
        
        # Load and prepare data
        data = load_data(ticker, start_date, end_date)
        if data is None or data.empty:
            raise ValueError(f"No data available for {ticker}")
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "status": "Adding technical indicators"})
        
        # Add technical indicators
        data_with_indicators = add_technical_indicators(data)
        
        self.update_state(state="PROGRESS", meta={"progress": 50, "status": "Training ML models"})
        
        # Create and train ensemble predictor
        predictor = EnsemblePredictor()
        training_results = predictor.train(data_with_indicators)
        
        if training_results.get("prophet", {}).get("status") != "success" and \
           training_results.get("lstm", {}).get("status") != "success":
            raise ValueError("Both ML models failed to train")
        
        self.update_state(state="PROGRESS", meta={"progress": 80, "status": "Generating predictions"})
        
        # Generate predictions
        predictions = predictor.predict(data_with_indicators, days=years_to_predict * 365)
        
        self.update_state(state="PROGRESS", meta={"progress": 95, "status": "Finalizing results"})
        
        # Prepare result
        result = {
            "ticker": ticker,
            "training_results": training_results,
            "predictions": predictions,
            "model_confidence": predictor.get_model_confidence(),
            "data_points": len(data_with_indicators),
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"ML prediction task failed for {ticker}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "progress": 0}
        )
        raise

# Sentiment Analysis Tasks
@celery_app.task(bind=True, name="celery_tasks.sentiment_analysis")
def sentiment_analysis_task(self, ticker: str, sources: List[str] = None,
                          days_back: int = 7, include_trend: bool = True) -> Dict[str, Any]:
    """Run sentiment analysis task."""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Initializing sentiment analyzer"})
        
        # Initialize sentiment aggregator
        aggregator = SentimentAggregator()
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "status": "Analyzing news sentiment"})
        
        # Get comprehensive sentiment
        sentiment_data = asyncio.run(aggregator.get_comprehensive_sentiment(ticker))
        
        if include_trend:
            self.update_state(state="PROGRESS", meta={"progress": 70, "status": "Analyzing sentiment trend"})
            sentiment_data["trend"] = aggregator.get_sentiment_trend(ticker, days_back)
        
        self.update_state(state="PROGRESS", meta={"progress": 95, "status": "Finalizing results"})
        
        result = {
            "ticker": ticker,
            "sentiment_data": sentiment_data,
            "sources_analyzed": sources or ["news", "twitter"],
            "days_analyzed": days_back,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Sentiment analysis task failed for {ticker}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "progress": 0}
        )
        raise

# Data Fetching Tasks
@celery_app.task(name="celery_tasks.fetch_market_data")
def fetch_market_data_task(tickers: List[str] = None) -> Dict[str, Any]:
    """Fetch latest market data for tickers."""
    try:
        if not tickers:
            # Default to major indices
            tickers = ["^GSPC", "^DJI", "^IXIC", "AAPL", "GOOGL", "MSFT"]
        
        results = {}
        
        for ticker in tickers:
            try:
                # Get latest data
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d", interval="1m")
                
                if not data.empty:
                    latest = data.iloc[-1]
                    results[ticker] = {
                        "price": float(latest["Close"]),
                        "change": float(latest["Close"] - latest["Open"]),
                        "change_percent": float(((latest["Close"] - latest["Open"]) / latest["Open"]) * 100),
                        "volume": int(latest["Volume"]),
                        "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
                    }
                    
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                results[ticker] = {"error": str(e)}
        
        return {
            "tickers": tickers,
            "data": results,
            "fetched_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market data fetch task failed: {e}")
        raise

# Alert Processing Tasks
@celery_app.task(name="celery_tasks.process_alerts")
def process_alerts_task() -> Dict[str, Any]:
    """Process all active alerts."""
    try:
        # This would integrate with the alert system
        # For now, return mock results
        triggered_alerts = []
        
        # Get market data
        market_data = fetch_market_data_task()
        
        # Process alerts (mock implementation)
        for ticker, data in market_data.get("data", {}).items():
            if "error" not in data:
                # Mock alert processing logic
                if abs(data.get("change_percent", 0)) > 5:  # 5% change
                    triggered_alerts.append({
                        "ticker": ticker,
                        "type": "price_movement",
                        "value": data["change_percent"],
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        return {
            "processed_alerts": len(market_data.get("data", {})),
            "triggered_alerts": triggered_alerts,
            "processed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alert processing task failed: {e}")
        raise

# Batch Prediction Tasks
@celery_app.task(bind=True, name="celery_tasks.batch_prediction")
def batch_prediction_task(self, ticker_list: List[str], start_date: str, end_date: str,
                         years_to_predict: int = 1, backtest_days: int = 30) -> Dict[str, Any]:
    """Run batch predictions for multiple tickers."""
    try:
        results = {}
        total_tickers = len(ticker_list)
        
        for i, ticker in enumerate(ticker_list):
            progress = int((i / total_tickers) * 90) + 10
            self.update_state(
                state="PROGRESS",
                meta={"progress": progress, "status": f"Processing {ticker} ({i+1}/{total_tickers})"}
            )
            
            try:
                # Run prediction for each ticker
                prediction_result = ml_prediction_task.delay(
                    ticker, start_date, end_date, years_to_predict, backtest_days
                )
                
                # Wait for result (with timeout)
                result = prediction_result.get(timeout=300)  # 5 minutes timeout
                results[ticker] = result
                
            except Exception as e:
                logger.error(f"Batch prediction failed for {ticker}: {e}")
                results[ticker] = {"error": str(e)}
        
        return {
            "batch_id": self.request.id,
            "tickers": ticker_list,
            "results": results,
            "successful": len([r for r in results.values() if "error" not in r]),
            "failed": len([r for r in results.values() if "error" in r]),
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction task failed: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "progress": 0}
        )
        raise

# Portfolio Analysis Tasks
@celery_app.task(bind=True, name="celery_tasks.portfolio_analysis")
def portfolio_analysis_task(self, portfolio_id: int, user_id: int) -> Dict[str, Any]:
    """Analyze portfolio performance and risk."""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Loading portfolio data"})
        
        # Mock portfolio data - in real implementation, this would query the database
        portfolio_holdings = [
            {"ticker": "AAPL", "shares": 100, "average_cost": 150.0},
            {"ticker": "GOOGL", "shares": 50, "average_cost": 2500.0},
            {"ticker": "MSFT", "shares": 75, "average_cost": 300.0}
        ]
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "status": "Fetching current prices"})
        
        # Get current prices
        tickers = [h["ticker"] for h in portfolio_holdings]
        market_data = fetch_market_data_task(tickers)
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "status": "Calculating performance metrics"})
        
        # Calculate portfolio metrics
        total_value = 0
        total_cost = 0
        holdings_analysis = []
        
        for holding in portfolio_holdings:
            ticker = holding["ticker"]
            current_price = market_data.get("data", {}).get(ticker, {}).get("price", 0)
            
            if current_price > 0:
                market_value = holding["shares"] * current_price
                cost_basis = holding["shares"] * holding["average_cost"]
                gain_loss = market_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
                
                holdings_analysis.append({
                    "ticker": ticker,
                    "shares": holding["shares"],
                    "average_cost": holding["average_cost"],
                    "current_price": current_price,
                    "market_value": market_value,
                    "cost_basis": cost_basis,
                    "gain_loss": gain_loss,
                    "gain_loss_percent": gain_loss_pct
                })
                
                total_value += market_value
                total_cost += cost_basis
        
        portfolio_return = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        self.update_state(state="PROGRESS", meta={"progress": 90, "status": "Generating risk analysis"})
        
        # Mock risk analysis
        risk_analysis = {
            "portfolio_beta": 1.2,
            "volatility": 0.18,
            "sharpe_ratio": 1.45,
            "max_drawdown": -0.12,
            "var_95": 0.025  # 2.5% Value at Risk
        }
        
        return {
            "portfolio_id": portfolio_id,
            "user_id": user_id,
            "holdings": holdings_analysis,
            "summary": {
                "total_value": total_value,
                "total_cost": total_cost,
                "total_gain_loss": total_value - total_cost,
                "total_return_percent": portfolio_return
            },
            "risk_analysis": risk_analysis,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio analysis task failed: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "progress": 0}
        )
        raise

# Cleanup Tasks
@celery_app.task(name="celery_tasks.cleanup_old_results")
def cleanup_old_results_task() -> Dict[str, Any]:
    """Clean up old task results and cache entries."""
    try:
        from celery_app import redis_client
        
        # Clean up old task metrics (older than 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Get all task metrics keys
        task_metrics_keys = redis_client.keys("task_metrics:*")
        task_status_keys = redis_client.keys("task_status:*")
        
        cleaned_metrics = 0
        cleaned_status = 0
        
        # Clean old metrics
        for key in task_metrics_keys:
            timestamp = redis_client.hget(key, "timestamp")
            if timestamp:
                try:
                    task_time = datetime.fromisoformat(timestamp)
                    if task_time < cutoff_time:
                        redis_client.delete(key)
                        cleaned_metrics += 1
                except:
                    pass
        
        # Clean old status
        for key in task_status_keys:
            timestamp = redis_client.hget(key, "end_time")
            if timestamp:
                try:
                    task_time = datetime.fromisoformat(timestamp)
                    if task_time < cutoff_time:
                        redis_client.delete(key)
                        cleaned_status += 1
                except:
                    pass
        
        # Clean old price cache
        price_keys = redis_client.keys("price:*")
        cleaned_prices = 0
        
        for key in price_keys:
            ttl = redis_client.ttl(key)
            if ttl == -1:  # No expiration set
                redis_client.expire(key, 3600)  # Set 1 hour expiration
                cleaned_prices += 1
        
        return {
            "cleaned_metrics": cleaned_metrics,
            "cleaned_status": cleaned_status,
            "cleaned_prices": cleaned_prices,
            "cleanup_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise

# Report Generation Tasks
@celery_app.task(bind=True, name="celery_tasks.generate_report")
def generate_report_task(self, report_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Generate various types of reports."""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": f"Generating {report_type} report"})
        
        if report_type == "portfolio_performance":
            # Generate portfolio performance report
            portfolio_id = parameters.get("portfolio_id")
            user_id = parameters.get("user_id")
            
            portfolio_data = portfolio_analysis_task.delay(portfolio_id, user_id)
            portfolio_result = portfolio_data.get()
            
            self.update_state(state="PROGRESS", meta={"progress": 80, "status": "Finalizing report"})
            
            return {
                "report_type": report_type,
                "data": portfolio_result,
                "generated_at": datetime.utcnow().isoformat()
            }
        
        elif report_type == "market_summary":
            # Generate market summary report
            tickers = parameters.get("tickers", ["^GSPC", "^DJI", "^IXIC"])
            
            market_data = fetch_market_data_task(tickers)
            
            self.update_state(state="PROGRESS", meta={"progress": 80, "status": "Finalizing report"})
            
            return {
                "report_type": report_type,
                "data": market_data,
                "generated_at": datetime.utcnow().isoformat()
            }
        
        else:
            raise ValueError(f"Unknown report type: {report_type}")
        
    except Exception as e:
        logger.error(f"Report generation task failed: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "progress": 0}
        )
        raise

# Export all tasks
__all__ = [
    "ml_prediction_task",
    "sentiment_analysis_task", 
    "fetch_market_data_task",
    "process_alerts_task",
    "batch_prediction_task",
    "portfolio_analysis_task",
    "cleanup_old_results_task",
    "generate_report_task"
]

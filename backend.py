from datetime import date
from datetime import datetime, timezone
import os
import time
import uuid
import logging
import asyncio
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Header, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from limits.storage import MemoryStorage

from services import generate_prediction_bundle
from database import get_db, PredictionRecord, TaskRecord, PredictionRequest, PredictionResponse, TaskResponse
from sqlalchemy.orm import Session
from exceptions import (
    StockVisionException, DataValidationError, InsufficientDataError,
    DataSourceError, ModelTrainingError, BacktestError,
    TaskNotFoundError, TaskTimeoutError, RateLimitExceededError
)

app = FastAPI(title="StockVision Backend API", version="1.0.0")
logger = logging.getLogger("stockvision.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_id=%s method=%s path=%s status=500", request_id, request.method, request.url.path)
        raise
    duration_ms = int((time.time() - start_time) * 1000)
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"status_code": exc.status_code, "message": str(exc.detail), "path": request.url.path}},
    )


@app.exception_handler(StockVisionException)
async def stockvision_exception_handler(request: Request, exc: StockVisionException):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "path": request.url.path,
                "details": exc.details
            }
        },
    )


def validate_api_key(incoming_api_key: str | None):
    configured_api_key = os.getenv("STOCKVISION_API_KEY", "").strip()
    if configured_api_key and incoming_api_key != configured_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")


def sanitize_nested_nans(payload):
    if isinstance(payload, dict):
        return {k: sanitize_nested_nans(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [sanitize_nested_nans(v) for v in payload]
    # NaN is the only value that is not equal to itself.
    if isinstance(payload, float) and payload != payload:
        return None
    return payload


@app.get("/")
@limiter.limit("10/minute")
def root(request: Request):
    return {
        "service": "stockvision-backend",
        "version": app.version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict?ticker=AAPL&start_date=2024-01-01&end_date=2024-03-01"
        },
        "docs": "/docs"
    }


@app.get("/health")
@limiter.limit("30/minute")
def health(request: Request):
    return {
        "status": "ok",
        "service": "stockvision-backend",
        "version": app.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "api_key_required": bool(os.getenv("STOCKVISION_API_KEY", "").strip()),
    }


async def run_prediction_task(
    task_id: str,
    request_data: PredictionRequest,
    db: Session
) -> None:
    """Background task to run prediction and store results."""
    try:
        # Update task status to running
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            logger.error(f"Task {task_id} not found")
            return
        
        task.status = "running"
        task.progress = 25
        db.commit()
        
        # Generate prediction
        bundle = generate_prediction_bundle(
            ticker=request_data.ticker.upper(),
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            years_to_predict=request_data.years_to_predict,
            backtest_days=request_data.backtest_days,
            signal_threshold_pct=request_data.signal_threshold_pct,
        )
        
        task.progress = 75
        db.commit()
        
        # Prepare response data
        latest = bundle["tech_data"].iloc[-1]
        indicator_cols = ["Date", "Close", "SMA_20", "EMA_20", "RSI_14", "Volatility_30"]
        
        response_payload = {
            "request_id": task_id,
            "ticker": request_data.ticker.upper(),
            "period": bundle["period"],
            "metrics": bundle["metrics"],
            "backtest": bundle["backtest"],
            "signal_info": bundle["signal_info"],
            "next_day_prediction": bundle["next_day_prediction"],
            "training_rows": int(len(bundle["df_train"])),
            "historical": bundle["new_data"].to_dict(orient="records"),
            "forecast": bundle["new_forecast"].to_dict(orient="records"),
            "stats": bundle["stats_data"].describe().to_dict(),
            "indicators_latest": {
                "SMA_20": None if latest["SMA_20"] != latest["SMA_20"] else float(latest["SMA_20"]),
                "EMA_20": None if latest["EMA_20"] != latest["EMA_20"] else float(latest["EMA_20"]),
                "RSI_14": None if latest["RSI_14"] != latest["RSI_14"] else float(latest["RSI_14"]),
                "Volatility_30": None if latest["Volatility_30"] != latest["Volatility_30"] else float(latest["Volatility_30"]),
            },
            "indicators_table": bundle["tech_data"][indicator_cols].tail(60).to_dict(orient="records"),
        }
        
        # Store in database
        prediction_record = PredictionRecord(
            ticker=request_data.ticker.upper(),
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            years_to_predict=request_data.years_to_predict,
            backtest_days=request_data.backtest_days,
            signal_threshold_pct=request_data.signal_threshold_pct,
            mae=bundle["metrics"]["mae"],
            rmse=bundle["metrics"]["rmse"],
            mape=bundle["metrics"]["mape"],
            signal=bundle["signal_info"]["signal"],
            expected_return_pct=bundle["signal_info"]["expected_return_percent"],
            next_day_prediction=bundle["next_day_prediction"],
            backtest_status=bundle["backtest"]["status"],
            request_id=task_id,
            historical_data=str(bundle["new_data"].to_dict(orient="records")),
            forecast_data=str(bundle["new_forecast"].to_dict(orient="records")),
            tech_indicators=str(bundle["tech_data"][indicator_cols].tail(60).to_dict(orient="records")),
            backtest_metrics=str(bundle["backtest"])
        )
        
        db.add(prediction_record)
        
        # Update task as completed
        task.status = "completed"
        task.progress = 100
        task.result = str(response_payload)
        db.commit()
        
        logger.info(f"Prediction task {task_id} completed successfully")
        
    except Exception as exc:
        logger.exception(f"Prediction task {task_id} failed: {str(exc)}")
        task.status = "failed"
        task.error_message = str(exc)
        db.commit()


@app.post("/predict/async")
@limiter.limit("5/minute")
def predict_async(
    request: Request,
    request_data: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    x_api_key: str | None = Header(default=None),
):
    """Start async prediction task."""
    validate_api_key(x_api_key)
    
    # Create task record
    task_id = str(uuid.uuid4())
    task = TaskRecord(
        id=task_id,
        ticker=request_data.ticker.upper(),
        status="pending"
    )
    db.add(task)
    db.commit()
    
    # Start background task
    background_tasks.add_task(run_prediction_task, task_id, request_data, db)
    
    return {"task_id": task_id, "status": "pending"}


@app.get("/predict/async/{task_id}")
@limiter.limit("20/minute")
def get_task_status(
    request: Request,
    task_id: str,
    db: Session = Depends(get_db)
):
    """Get async prediction task status."""
    task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    response = TaskResponse(
        task_id=task.id,
        status=task.status,
        progress=task.progress,
        result=eval(task.result) if task.result else None,
        error_message=task.error_message,
        created_at=task.created_at,
        updated_at=task.updated_at
    )
    
    return response


@app.get("/predict")
@limiter.limit("10/minute")
def predict(
    request: Request,
    ticker: str = Query(..., min_length=1),
    start_date: date = Query(...),
    end_date: date = Query(...),
    years_to_predict: int = Query(1, ge=1, le=5),
    backtest_days: int = Query(30, ge=15, le=90),
    signal_threshold_pct: float = Query(2.0, ge=0.5, le=5.0),
    x_api_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    """Synchronous prediction endpoint (original behavior)."""
    validate_api_key(x_api_key)
    
    try:
        bundle = generate_prediction_bundle(
            ticker=ticker.upper(),
            start_date=start_date,
            end_date=end_date,
            years_to_predict=years_to_predict,
            backtest_days=backtest_days,
            signal_threshold_pct=signal_threshold_pct,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}") from exc

    latest = bundle["tech_data"].iloc[-1]
    indicator_cols = ["Date", "Close", "SMA_20", "EMA_20", "RSI_14", "Volatility_30"]

    response_payload = {
        "request_id": request.headers.get("x-request-id", ""),
        "ticker": ticker.upper(),
        "period": bundle["period"],
        "metrics": bundle["metrics"],
        "backtest": bundle["backtest"],
        "signal_info": bundle["signal_info"],
        "next_day_prediction": bundle["next_day_prediction"],
        "training_rows": int(len(bundle["df_train"])),
        "historical": bundle["new_data"].to_dict(orient="records"),
        "forecast": bundle["new_forecast"].to_dict(orient="records"),
        "stats": bundle["stats_data"].describe().to_dict(),
        "indicators_latest": {
            "SMA_20": None if latest["SMA_20"] != latest["SMA_20"] else float(latest["SMA_20"]),
            "EMA_20": None if latest["EMA_20"] != latest["EMA_20"] else float(latest["EMA_20"]),
            "RSI_14": None if latest["RSI_14"] != latest["RSI_14"] else float(latest["RSI_14"]),
            "Volatility_30": None if latest["Volatility_30"] != latest["Volatility_30"] else float(latest["Volatility_30"]),
        },
        "indicators_table": bundle["tech_data"][indicator_cols].tail(60).to_dict(orient="records"),
    }
    
    # Store in database
    prediction_record = PredictionRecord(
        ticker=ticker.upper(),
        start_date=start_date,
        end_date=end_date,
        years_to_predict=years_to_predict,
        backtest_days=backtest_days,
        signal_threshold_pct=signal_threshold_pct,
        mae=bundle["metrics"]["mae"],
        rmse=bundle["metrics"]["rmse"],
        mape=bundle["metrics"]["mape"],
        signal=bundle["signal_info"]["signal"],
        expected_return_pct=bundle["signal_info"]["expected_return_percent"],
        next_day_prediction=bundle["next_day_prediction"],
        backtest_status=bundle["backtest"]["status"],
        request_id=request.headers.get("x-request-id", ""),
        historical_data=str(bundle["new_data"].to_dict(orient="records")),
        forecast_data=str(bundle["new_forecast"].to_dict(orient="records")),
        tech_indicators=str(bundle["tech_data"][indicator_cols].tail(60).to_dict(orient="records")),
        backtest_metrics=str(bundle["backtest"])
    )
    
    db.add(prediction_record)
    db.commit()
    
    return sanitize_nested_nans(response_payload)


@app.get("/predictions/history")
@limiter.limit("20/minute")
def get_prediction_history(
    request: Request,
    ticker: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get historical predictions."""
    query = db.query(PredictionRecord)
    
    if ticker:
        query = query.filter(PredictionRecord.ticker == ticker.upper())
    
    predictions = query.order_by(PredictionRecord.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": pred.id,
            "ticker": pred.ticker,
            "start_date": pred.start_date.isoformat(),
            "end_date": pred.end_date.isoformat(),
            "mae": pred.mae,
            "rmse": pred.rmse,
            "mape": pred.mape,
            "signal": pred.signal,
            "expected_return_pct": pred.expected_return_pct,
            "created_at": pred.created_at.isoformat()
        }
        for pred in predictions
    ]

from datetime import date
from datetime import datetime, timezone
import os
import time
import uuid
import logging

from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services import generate_prediction_bundle


app = FastAPI(title="StockVision Backend API", version="1.0.0")
logger = logging.getLogger("stockvision.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "stockvision-backend",
        "version": app.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "api_key_required": bool(os.getenv("STOCKVISION_API_KEY", "").strip()),
    }


@app.get("/predict")
def predict(
    request: Request,
    ticker: str = Query(..., min_length=1),
    start_date: date = Query(...),
    end_date: date = Query(...),
    years_to_predict: int = Query(1, ge=1, le=5),
    backtest_days: int = Query(30, ge=15, le=90),
    signal_threshold_pct: float = Query(2.0, ge=0.5, le=5.0),
    x_api_key: str | None = Header(default=None),
):
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
    return sanitize_nested_nans(response_payload)

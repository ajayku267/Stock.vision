from datetime import date

from fastapi import FastAPI, HTTPException, Query

from services import generate_prediction_bundle


app = FastAPI(title="StockVision Backend API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(
    ticker: str = Query(..., min_length=1),
    start_date: date = Query(...),
    end_date: date = Query(...),
    years_to_predict: int = Query(1, ge=1, le=5),
    backtest_days: int = Query(30, ge=15, le=90),
    signal_threshold_pct: float = Query(2.0, ge=0.5, le=5.0),
):
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

    return {
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

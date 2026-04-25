import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi.testclient import TestClient

import backend


def _mock_bundle():
    tech = pd.DataFrame(
        [
            {
                "Date": "2024-01-01",
                "Close": 100.0,
                "SMA_20": 99.5,
                "EMA_20": 99.8,
                "RSI_14": 55.2,
                "Volatility_30": 0.21,
            }
        ]
    )
    return {
        "period": 365,
        "metrics": {"mae": 1.1, "rmse": 2.2, "mape": 3.3},
        "backtest": {"status": "ok", "test_days": 30, "mae": 1.2, "rmse": 2.3, "mape_percent": 3.4},
        "signal_info": {"signal": "BULLISH", "expected_return_percent": 2.1},
        "next_day_prediction": 105.0,
        "df_train": pd.DataFrame({"ds": pd.to_datetime(["2024-01-01"]), "y": [100.0]}),
        "new_data": pd.DataFrame([{"Date": "2024-01-01", "Open": 99.0, "Close": 100.0}]),
        "new_forecast": pd.DataFrame(
            [{"Date": "2024-01-02", "Close": 105.0, "Close Lower": 102.0, "Close Upper": 108.0}]
        ),
        "stats_data": pd.DataFrame([{"Open": 99.0, "Close": 100.0}]),
        "tech_data": tech,
    }


def test_health_endpoint():
    client = TestClient(backend.app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "stockvision-backend"


def test_predict_endpoint_success(monkeypatch):
    monkeypatch.delenv("STOCKVISION_API_KEY", raising=False)
    monkeypatch.setattr(backend, "generate_prediction_bundle", lambda **kwargs: _mock_bundle())

    client = TestClient(backend.app)
    response = client.get(
        "/predict",
        params={
            "ticker": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "years_to_predict": 1,
            "backtest_days": 30,
            "signal_threshold_pct": 2.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["signal_info"]["signal"] == "BULLISH"
    assert "historical" in payload


def test_predict_requires_api_key(monkeypatch):
    monkeypatch.setenv("STOCKVISION_API_KEY", "secret-key")
    monkeypatch.setattr(backend, "generate_prediction_bundle", lambda **kwargs: _mock_bundle())

    client = TestClient(backend.app)
    response = client.get(
        "/predict",
        params={
            "ticker": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "years_to_predict": 1,
            "backtest_days": 30,
            "signal_threshold_pct": 2.0,
        },
    )
    assert response.status_code == 401
    assert "error" in response.json()

    ok_response = client.get(
        "/predict",
        params={
            "ticker": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "years_to_predict": 1,
            "backtest_days": 30,
            "signal_threshold_pct": 2.0,
        },
        headers={"X-API-Key": "secret-key"},
    )
    assert ok_response.status_code == 200

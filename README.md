# Stock.Vision

An interview-ready stock forecasting platform built with a **Streamlit frontend** and a **FastAPI backend**.

It combines time-series forecasting (Prophet), technical indicators, backtesting, and API-driven prediction mode in one polished project.

## Why This Project Stands Out

- Full-stack architecture: UI + backend API + shared service layer
- Practical forecasting pipeline with validation and error handling
- Technical indicators for market context (SMA, EMA, RSI, volatility)
- Backtest metrics (MAE, RMSE, MAPE) for model reliability
- Downloadable datasets and interview-focused presentation tab

## Tech Stack

- Python 3.14
- Streamlit (dashboard frontend)
- FastAPI + Uvicorn (backend API)
- Prophet (forecasting)
- yfinance (market data)
- pandas, scikit-learn, plotly

## Project Structure

```text
Stock.vision/
├── main.py            # Streamlit app (frontend)
├── backend.py         # FastAPI service (backend)
├── services.py        # Shared forecasting/business logic
├── requirements.txt   # Pinned dependencies
└── README.md
```

## Core Features

### Frontend (Streamlit)

- Interactive dashboard with multiple tabs:
  - Dataframes
  - Plots
  - Statistics
  - Forecasting
  - Comparison
  - Interview View
- KPI cards: MAE, RMSE, MAPE, signal, expected return
- CSV download buttons for historical + forecast data
- Toggle between:
  - **Local mode** (direct forecasting in app)
  - **Backend API mode** (frontend consumes FastAPI)

### Backend (FastAPI)

- `GET /health` for service status
- `GET /predict` for full prediction payload:
  - historical data
  - forecasted values
  - model metrics
  - backtest results
  - technical indicators
  - bullish/neutral/bearish signal
- Optional API key security via `STOCKVISION_API_KEY` (`X-API-Key` header)
- Request logging middleware with request IDs for observability

### Forecasting + Quant Logic

- Prophet model training with clean data preparation
- Technical indicators:
  - `SMA_20`
  - `EMA_20`
  - `RSI_14`
  - `Volatility_30`
- Holdout backtesting on recent N days
- Rule-based signal engine using configurable return threshold

## Quick Start

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Start backend API

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

### 3) Start Streamlit frontend

```bash
python -m streamlit run main.py --server.port 8501
```

### 4) Open app

- Frontend: `http://localhost:8501`
- Backend docs: `http://127.0.0.1:8000/docs`

## Recommended Hosting (Chosen)

For your goal ("open a link and present without editor"), the best fit is:

- **Render full-stack deployment** using `render.yaml`
- Hosts both:
  - `stockvision-backend` (FastAPI)
  - `stockvision-frontend` (Streamlit)
- Frontend auto-connects to backend via environment variables

### Deploy on Render

1. Push this repo to GitHub.
2. In Render: **New +** -> **Blueprint**.
3. Select your repo (Render will detect `render.yaml`).
4. Click **Apply**.
5. Wait for both services to become live.
6. Open the frontend Render URL and demo directly.

### Hosting Notes

- `runtime.txt` pins Python runtime for consistent builds.
- CORS is enabled in backend so frontend/backend can communicate across domains.
- Backend API docs will be available at:
  - `https://<your-backend-service>/docs`

## API Example

```bash
curl "http://127.0.0.1:8000/predict?ticker=AAPL&start_date=2024-01-01&end_date=2024-06-01&years_to_predict=1&backtest_days=30&signal_threshold_pct=2.0"
```

If API key is enabled:

```bash
curl -H "X-API-Key: your-secret-key" "http://127.0.0.1:8000/predict?ticker=AAPL&start_date=2024-01-01&end_date=2024-06-01&years_to_predict=1&backtest_days=30&signal_threshold_pct=2.0"
```

## Production-Style Environment Variables

- `STOCKVISION_USE_BACKEND_DEFAULT=true`
- `STOCKVISION_BACKEND_URL=https://<backend-domain>`
- `STOCKVISION_API_KEY=<strong-secret>` (optional backend protection)

## Testing & CI

- Backend tests live in `tests/test_backend.py`
- CI workflow in `.github/workflows/ci.yml` runs:
  - dependency install
  - syntax validation
  - automated tests (`pytest`)

## Suggested Interview Demo Flow

1. Show architecture (frontend vs backend mode)
2. Run a ticker forecast and explain model metrics
3. Show backtesting for reliability
4. Explain indicator-driven signal output
5. Switch to API mode and prove backend integration
6. Open `/docs` to show production-style API contract

## Future Improvements

- Add authentication + rate limiting on API endpoints
- Persist historical/forecast data in PostgreSQL
- Add async task queue for long-running model jobs
- Add CI pipeline + unit/integration test suite
- Add Docker + docker-compose for one-command setup

## Notes

- This project is for educational and interview demonstration purposes.
- Forecasts are model-based estimates, **not financial advice**.

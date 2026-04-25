import streamlit as st
from plotly import graph_objs as go
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests

@st.cache_data
def load_data(ticker, start, end):
    """
    Load historical stock price data from Yahoo Finance.

    Parameters:
    - ticker (str): Stock symbol (e.g., AAPL).
    - start (str): Start date in the format 'YYYY-MM-DD'.
    - end (str): End date in the format 'YYYY-MM-DD'.

    Returns:
    - data (pd.DataFrame): DataFrame containing historical stock price data.
    """
    try:
        data = yf.download(ticker, start, end)
        # yfinance may return MultiIndex columns on newer versions.
        # Flatten to simple column names so downstream code gets 1-D Series.
        if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

def plot_data(data):
    """
    Plot historical stock price data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing historical stock price data.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.update_layout(title_text="Stock Prices Over Time", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, width="stretch")

def plot_multiple_data(data, stock_names):
    """
    Plot forecasted stock prices for multiple stocks.

    Parameters:
    - data (list): List of DataFrames containing forecasted stock price data.
    - stock_names (list): List of stock names corresponding to the forecasted data.
    """
    fig = go.Figure()
    for i, stock_data in enumerate(data):
        fig.add_trace(go.Scatter(x=stock_data['ds'], y=stock_data['yhat'], name=f"yhat - {stock_names[i]}"))
    fig.update_layout(title_text="Stock Prices Over Time", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, width="stretch")

def plot_volume(data):
    """
    Plot historical stock volume data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing historical stock volume data.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], name="stock_volume"))
    fig.update_layout(title_text="Stock Volume Over Time", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, width="stretch")


def prepare_training_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data[["Date", "Close"]].copy()
    frame = frame.rename(columns={"Date": "ds", "Close": "y"})
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame.dropna(subset=["ds", "y"])
    return frame


def prepare_forecast_table(forecast: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "additive_terms",
        "additive_terms_lower",
        "additive_terms_upper",
        "weekly",
        "weekly_lower",
        "weekly_upper",
        "yearly",
        "yearly_lower",
        "yearly_upper",
        "multiplicative_terms",
        "multiplicative_terms_lower",
        "multiplicative_terms_upper",
    ]
    slim = forecast.drop(columns=drop_cols, errors="ignore").copy()
    slim = slim.rename(
        columns={
            "ds": "Date",
            "yhat": "Close",
            "yhat_lower": "Close Lower",
            "yhat_upper": "Close Upper",
            "trend": "Trend",
            "trend_lower": "Trend Lower",
            "trend_upper": "Trend Upper",
        }
    )
    return slim


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    enriched["SMA_20"] = enriched["Close"].rolling(window=20).mean()
    enriched["EMA_20"] = enriched["Close"].ewm(span=20, adjust=False).mean()

    delta = enriched["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    enriched["RSI_14"] = 100 - (100 / (1 + rs))

    enriched["Daily_Return"] = enriched["Close"].pct_change()
    enriched["Volatility_30"] = enriched["Daily_Return"].rolling(window=30).std() * np.sqrt(252)
    return enriched


def run_prophet_backtest(df_train: pd.DataFrame, test_days: int = 30) -> dict:
    if len(df_train) <= test_days + 20:
        return {"status": "insufficient_data"}

    train_slice = df_train.iloc[:-test_days].copy()
    test_slice = df_train.iloc[-test_days:].copy()

    model = Prophet()
    model.fit(train_slice)
    future = model.make_future_dataframe(periods=test_days)
    forecast = model.predict(future)[["ds", "yhat"]]
    test_pred = forecast[forecast["ds"].isin(test_slice["ds"])].copy()

    merged = test_slice.merge(test_pred, on="ds", how="inner")
    if merged.empty:
        return {"status": "prediction_alignment_failed"}

    mae = mean_absolute_error(merged["y"], merged["yhat"])
    rmse = mean_squared_error(merged["y"], merged["yhat"]) ** 0.5
    mape = (
        abs((merged["y"] - merged["yhat"]) / merged["y"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean()
        * 100
    )
    return {
        "status": "ok",
        "test_days": int(test_days),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_percent": float(mape),
    }


def derive_forecast_signal(last_close: float, next_close: float, threshold: float = 0.02) -> dict:
    if last_close <= 0:
        return {"signal": "UNKNOWN", "expected_return_percent": 0.0}

    expected_return = (next_close - last_close) / last_close
    if expected_return >= threshold:
        signal = "BULLISH"
    elif expected_return <= -threshold:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"
    return {
        "signal": signal,
        "expected_return_percent": float(expected_return * 100),
    }


def generate_prediction_bundle(
    ticker: str,
    start_date,
    end_date,
    years_to_predict: int,
    backtest_days: int,
    signal_threshold_pct: float,
):
    data = load_data(ticker, start_date, end_date)
    if data is None or data.empty:
        raise ValueError("No data returned for the selected stock/date range.")

    required_cols = {"Date", "Close", "Open", "Volume"}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")

    if start_date >= end_date:
        raise ValueError("Start date must be before end date.")

    df_train = prepare_training_frame(data)
    if len(df_train) < 30:
        raise ValueError("Not enough clean rows to train forecast model.")

    period = years_to_predict * 365
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    forecast_full = model.predict(future)

    end_date_datetime = pd.to_datetime(pd.to_datetime(end_date).strftime("%Y-%m-%d"))
    forecast = forecast_full[forecast_full["ds"] >= end_date_datetime].copy()

    train_pred = model.predict(df_train[["ds"]])[["ds", "yhat"]]
    eval_df = df_train.merge(train_pred, on="ds", how="inner")
    mae = mean_absolute_error(eval_df["y"], eval_df["yhat"])
    rmse = mean_squared_error(eval_df["y"], eval_df["yhat"]) ** 0.5
    mape = (
        abs((eval_df["y"] - eval_df["yhat"]) / eval_df["y"])
        .replace([float("inf"), -float("inf")], pd.NA)
        .dropna()
        .mean()
    ) * 100

    tech_data = add_technical_indicators(data)
    backtest = run_prophet_backtest(df_train, test_days=backtest_days)
    next_day_prediction = (
        forecast.sort_values("ds").iloc[0]["yhat"] if not forecast.empty else float(df_train["y"].iloc[-1])
    )
    signal_info = derive_forecast_signal(
        last_close=float(df_train["y"].iloc[-1]),
        next_close=float(next_day_prediction),
        threshold=signal_threshold_pct / 100.0,
    )

    return {
        "data": data,
        "df_train": df_train,
        "forecast": forecast,
        "new_data": data.drop(columns=["Adj Close", "Volume"], errors="ignore"),
        "new_forecast": prepare_forecast_table(forecast),
        "stats_data": data.drop(columns=["Date", "Adj Close", "Volume"], errors="ignore"),
        "tech_data": tech_data,
        "model": model,
        "period": period,
        "metrics": {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)},
        "backtest": backtest,
        "signal_info": signal_info,
        "next_day_prediction": float(next_day_prediction),
    }


def fetch_backend_prediction(
    base_url: str,
    ticker: str,
    start_date,
    end_date,
    years_to_predict: int,
    backtest_days: int,
    signal_threshold_pct: float,
):
    response = requests.get(
        f"{base_url.rstrip('/')}/predict",
        params={
            "ticker": ticker,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "years_to_predict": years_to_predict,
            "backtest_days": backtest_days,
            "signal_threshold_pct": signal_threshold_pct,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
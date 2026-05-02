import streamlit as st
from plotly import graph_objs as go
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date
import cache
from data_extractors import data_extractor, real_time_stream
from news_aggregator import news_aggregator

@st.cache_data
def load_data(ticker: str, start: Union[str, date], end: Union[str, date]) -> Optional[pd.DataFrame]:
    """
    Load historical stock price data from Yahoo Finance.

    Parameters:
    - ticker (str): Stock symbol (e.g., AAPL).
    - start (str or date): Start date in the format 'YYYY-MM-DD' or a date object.
    - end (str or date): End date in the format 'YYYY-MM-DD' or a date object.

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

def plot_data(data: pd.DataFrame) -> None:
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

def plot_multiple_data(data: List[pd.DataFrame], stock_names: List[str]) -> None:
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

def plot_volume(data: pd.DataFrame) -> None:
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


def run_prophet_backtest(df_train: pd.DataFrame, test_days: int = 30) -> Dict[str, Any]:
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


def derive_forecast_signal(last_close: float, next_close: float, threshold: float = 0.02) -> Dict[str, Any]:
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
    start_date: Union[str, date],
    end_date: Union[str, date],
    years_to_predict: int,
    backtest_days: int,
    signal_threshold_pct: float,
) -> Dict[str, Any]:
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


def fetch_backend_prediction(ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch prediction from backend API."""
    backend_url = config.get("backend_url", "http://localhost:8001")
    
    try:
        response = requests.post(
            f"{backend_url}/predict",
            json={
                "ticker": ticker,
                "start_date": config["start_date"],
                "end_date": config["end_date"],
                "years_to_predict": config["years_to_predict"],
                "backtest_days": config["backtest_days"],
                "signal_threshold_pct": config["signal_threshold_pct"]
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching backend prediction: {str(e)}")
        return {"error": str(e)}

# Real-time data extraction functions
def get_real_time_stock_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Get real-time stock data from multiple sources."""
    try:
        stock_data = data_extractor.get_stock_data(ticker)
        if stock_data:
            return {
                "ticker": stock_data.ticker,
                "company_name": stock_data.company_name,
                "current_price": stock_data.current_price,
                "change": stock_data.change,
                "change_percent": stock_data.change_percent,
                "volume": stock_data.volume,
                "market_cap": stock_data.market_cap,
                "pe_ratio": stock_data.pe_ratio,
                "dividend_yield": stock_data.dividend_yield,
                "high_52w": stock_data.high_52w,
                "low_52w": stock_data.low_52w,
                "avg_volume": stock_data.avg_volume,
                "beta": stock_data.beta,
                "eps": stock_data.eps,
                "revenue": stock_data.revenue,
                "timestamp": stock_data.timestamp.isoformat()
            }
        return None
    except Exception as e:
        st.error(f"Error fetching real-time data for {ticker}: {str(e)}")
        return None

def get_multiple_real_time_data(tickers: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Get real-time data for multiple stocks."""
    results = {}
    for ticker in tickers:
        results[ticker] = get_real_time_stock_data(ticker)
    return results

def get_market_overview() -> Dict[str, Any]:
    """Get market overview with indices data."""
    try:
        overview = data_extractor.get_market_overview()
        return overview
    except Exception as e:
        st.error(f"Error fetching market overview: {str(e)}")
        return {}

def get_financial_news(ticker: str = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Get latest financial news."""
    try:
        if ticker:
            news = news_aggregator.get_news_by_ticker(ticker, hours_back=24)
        else:
            news = news_aggregator.get_all_news(limit_per_source=limit//4)
        
        # Convert to dict format
        news_list = []
        for article in news[:limit]:
            news_list.append({
                "title": article.title,
                "url": article.url,
                "source": article.source,
                "published_date": article.published_date.isoformat(),
                "summary": article.summary,
                "tickers": article.tickers,
                "author": article.author
            })
        
        return news_list
    except Exception as e:
        st.error(f"Error fetching financial news: {str(e)}")
        return []

def get_trending_stocks() -> Dict[str, int]:
    """Get trending stocks based on news mentions."""
    try:
        trending = news_aggregator.get_trending_stocks(hours_back=24)
        return trending
    except Exception as e:
        st.error(f"Error fetching trending stocks: {str(e)}")
        return {}

def get_market_sentiment() -> Dict[str, Any]:
    """Get overall market sentiment from news."""
    try:
        sentiment = news_aggregator.get_market_sentiment(hours_back=24)
        return sentiment
    except Exception as e:
        st.error(f"Error fetching market sentiment: {str(e)}")
        return {}

def plot_real_time_chart(ticker: str, data: Dict[str, Any]) -> None:
    """Plot real-time stock data with current price and indicators."""
    if not data:
        st.error("No data available for plotting")
        return
    
    # Create a simple price chart with current price
    fig = go.Figure()
    
    # Add current price point
    fig.add_trace(go.Scatter(
        x=[data["timestamp"]],
        y=[data["current_price"]],
        mode='markers+text',
        marker=dict(size=15, color='green' if data["change"] > 0 else 'red'),
        text=[f'${data["current_price"]:.2f}'],
        textposition="top center",
        name=f'{ticker} Current Price'
    ))
    
    # Add 52-week range if available
    if data.get("high_52w") and data.get("low_52w"):
        fig.add_hline(y=data["high_52w"], line_dash="dash", line_color="red", 
                     annotation_text=f"52W High: ${data['high_52w']:.2f}")
        fig.add_hline(y=data["low_52w"], line_dash="dash", line_color="green",
                     annotation_text=f"52W Low: ${data['low_52w']:.2f}")
    
    fig.update_layout(
        title=f"{data['company_name']} ({ticker}) - Real-Time Data",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")

def display_stock_metrics(data: Dict[str, Any]) -> None:
    """Display key stock metrics in a formatted way."""
    if not data:
        return
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${data['current_price']:.2f}",
            f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)",
            delta_color="normal" if data["change"] >= 0 else "inverse"
        )
    
    with col2:
        if data.get("market_cap"):
            market_cap = data["market_cap"]
            if market_cap > 1_000_000_000:
                st.metric("Market Cap", f"${market_cap/1_000_000_000:.1f}B")
            elif market_cap > 1_000_000:
                st.metric("Market Cap", f"${market_cap/1_000_000:.1f}M")
            else:
                st.metric("Market Cap", f"${market_cap:,.0f}")
    
    with col3:
        if data.get("pe_ratio"):
            st.metric("P/E Ratio", f"{data['pe_ratio']:.2f}")
    
    with col4:
        if data.get("volume"):
            volume = data["volume"]
            if volume > 1_000_000:
                st.metric("Volume", f"{volume/1_000_000:.1f}M")
            elif volume > 1_000:
                st.metric("Volume", f"{volume/1_000:.1f}K")
            else:
                st.metric("Volume", f"{volume:,}")
    
    # Additional metrics in expandable section
    with st.expander("Detailed Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            if data.get("dividend_yield"):
                st.metric("Dividend Yield", f"{data['dividend_yield']*100:.2f}%")
            if data.get("beta"):
                st.metric("Beta", f"{data['beta']:.2f}")
            if data.get("eps"):
                st.metric("EPS", f"${data['eps']:.2f}")
        
        with col2:
            if data.get("high_52w"):
                st.metric("52W High", f"${data['high_52w']:.2f}")
            if data.get("low_52w"):
                st.metric("52W Low", f"${data['low_52w']:.2f}")
            if data.get("avg_volume"):
                avg_vol = data["avg_volume"]
                if avg_vol > 1_000_000:
                    st.metric("Avg Volume", f"{avg_vol/1_000_000:.1f}M")
                else:
                    st.metric("Avg Volume", f"{avg_vol:,}")
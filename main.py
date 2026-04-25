import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from services import (
    load_data,
    plot_data,
    plot_multiple_data,
    plot_volume,
    prepare_training_frame,
    prepare_forecast_table,
    add_technical_indicators,
    run_prophet_backtest,
    derive_forecast_signal,
    generate_prediction_bundle,
    fetch_backend_prediction,
)


st.set_page_config(layout="wide", page_title="StockVision", page_icon="📈")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .app-hero {
        padding: 1.1rem 1.3rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(255,153,51,0.18), rgba(59,130,246,0.14));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .app-hero h1 {margin: 0; font-size: 1.8rem;}
    .app-hero p {margin: .35rem 0 0 0; opacity: .88;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<h1 style='text-align: center; font-size: 30px; margin-bottom: .2rem;'><b>Stock.</b><b style='color: orange'>Vision</b></h1>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Interview-ready analytics dashboard")

with st.sidebar.expander("Date Range", expanded=True):
    start_date = st.date_input("Start date", date(2018, 1, 1))
    end_date = st.date_input("End date", date.today())

stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMC", "TSLA", "AMZN", "NFLX", "NVDA", "AMD", "PYPL")
with st.sidebar.expander("Tickers", expanded=True):
    selected_stock = st.selectbox("Primary ticker", stocks)
    selected_stocks = st.multiselect("Comparison tickers", stocks, placeholder="Pick one or more")

with st.sidebar.expander("Forecast Settings", expanded=True):
    years_to_predict = st.slider("Years of prediction", 1, 5, help="Forecast horizon in years")
    backtest_days = st.slider("Backtest days", 15, 90, 30, help="How many recent days to reserve for backtesting")
    signal_threshold_pct = st.slider("Signal threshold (%)", 0.5, 5.0, 2.0, 0.5)
with st.sidebar.expander("Backend API", expanded=False):
    default_use_backend = os.getenv("STOCKVISION_USE_BACKEND_DEFAULT", "false").strip().lower() == "true"
    default_backend_url = os.getenv("STOCKVISION_BACKEND_URL", "http://127.0.0.1:8000").strip()
    if default_backend_url and not default_backend_url.startswith(("http://", "https://")):
        default_backend_url = f"https://{default_backend_url}"
    use_backend_api = st.toggle("Use backend API mode", value=default_use_backend)
    backend_url = st.text_input("Backend URL", value=default_backend_url)

period = years_to_predict * 365

st.markdown(
    """
    <div class="app-hero">
        <h1>Stock Forecasting App 📈</h1>
        <p><b>Stock.</b><b style="color: orange">Vision</b> helps analyze price behavior and forecast future trends.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_tab = option_menu(
    menu_title=None,
    options=["Dataframes", "Plots", "Statistics", "Forecasting", "Comparison", "Interview View"],
    icons=["table", "bar-chart", "calculator", "graph-up-arrow", "arrow-down-up", "briefcase"],
    menu_icon="📊",
    default_index=0,
    orientation="horizontal",
)
with st.spinner("Loading data..."):
    try:
        if use_backend_api:
            api_payload = fetch_backend_prediction(
                base_url=backend_url,
                ticker=selected_stock,
                start_date=start_date,
                end_date=end_date,
                years_to_predict=years_to_predict,
                backtest_days=backtest_days,
                signal_threshold_pct=signal_threshold_pct,
            )
            new_data = pd.DataFrame(api_payload["historical"])
            new_forecast = pd.DataFrame(api_payload["forecast"])
            stats_summary_df = pd.DataFrame(api_payload["stats"])
            tech_data = pd.DataFrame(api_payload["indicators_table"])
            data = new_data.copy()
            df_train = pd.DataFrame({"ds": pd.to_datetime(new_data["Date"]), "y": pd.to_numeric(new_data["Close"])})
            forecast = pd.DataFrame(
                {
                    "ds": pd.to_datetime(new_forecast["Date"]),
                    "yhat": pd.to_numeric(new_forecast["Close"]),
                    "yhat_lower": pd.to_numeric(new_forecast["Close Lower"]),
                    "yhat_upper": pd.to_numeric(new_forecast["Close Upper"]),
                }
            )
            model = None
            metrics = api_payload["metrics"]
            mae = metrics["mae"]
            rmse = metrics["rmse"]
            mape = metrics["mape"]
            backtest = api_payload["backtest"]
            signal_info = api_payload["signal_info"]
            next_day_prediction = api_payload["next_day_prediction"]
        else:
            bundle = generate_prediction_bundle(
                ticker=selected_stock,
                start_date=start_date,
                end_date=end_date,
                years_to_predict=years_to_predict,
                backtest_days=backtest_days,
                signal_threshold_pct=signal_threshold_pct,
            )
            data = bundle["data"]
            df_train = bundle["df_train"]
            forecast = bundle["forecast"]
            new_data = bundle["new_data"]
            new_forecast = bundle["new_forecast"]
            stats_data = bundle["stats_data"]
            tech_data = bundle["tech_data"]
            model = bundle["model"]
            mae = bundle["metrics"]["mae"]
            rmse = bundle["metrics"]["rmse"]
            mape = bundle["metrics"]["mape"]
            backtest = bundle["backtest"]
            signal_info = bundle["signal_info"]
            next_day_prediction = bundle["next_day_prediction"]
            stats_summary_df = stats_data.describe()
    except Exception as exc:
        st.error(f"Prediction pipeline failed: {str(exc)}")
        st.stop()

top_c1, top_c2, top_c3, top_c4 = st.columns(4)
top_c1.metric("Ticker", selected_stock)
top_c2.metric("Training Rows", f"{len(df_train):,}")
top_c3.metric("Forecast Horizon", f"{years_to_predict} yr")
top_c4.metric("MAPE", f"{mape:.2f}%")

if selected_tab == "Dataframes":
    st.markdown("<h2><span style='color: orange;'>{}</span> Historical Data</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section displays historical stock price data for {} from {} to {}.".format(selected_stock, start_date, end_date))

    st.dataframe(new_data, width="stretch")
    st.download_button(
        "Download historical data CSV",
        data=new_data.to_csv(index=False),
        file_name=f"{selected_stock.lower()}_historical.csv",
        mime="text/csv",
    )

    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Data</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section displays the forecasted stock price data for {} using the Prophet model from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))

    st.dataframe(new_forecast, width="stretch")
    st.download_button(
        "Download forecast data CSV",
        data=new_forecast.to_csv(index=False),
        file_name=f"{selected_stock.lower()}_forecast.csv",
        mime="text/csv",
    )

if selected_tab == "Plots":

    plot_data(data)

    plot_volume(data)


if selected_tab == "Statistics":
    st.markdown("<h2><span style='color: orange;'>Descriptive </span>Statistics</h2>", unsafe_allow_html=True)
    st.write("This section provides descriptive statistics for the selected stock.")

 
    st.table(stats_summary_df)

    st.markdown("### Technical Indicators (Latest)")
    latest = tech_data.iloc[-1]
    ind1, ind2, ind3, ind4 = st.columns(4)
    ind1.metric("SMA 20", f"{latest['SMA_20']:.2f}" if pd.notna(latest["SMA_20"]) else "N/A")
    ind2.metric("EMA 20", f"{latest['EMA_20']:.2f}" if pd.notna(latest["EMA_20"]) else "N/A")
    ind3.metric("RSI 14", f"{latest['RSI_14']:.2f}" if pd.notna(latest["RSI_14"]) else "N/A")
    ind4.metric("Volatility 30", f"{latest['Volatility_30']:.2%}" if pd.notna(latest["Volatility_30"]) else "N/A")

    indicator_cols = ["Date", "Close", "SMA_20", "EMA_20", "RSI_14", "Volatility_30"]
    st.dataframe(tech_data[indicator_cols].tail(60), width="stretch")

if selected_tab == "Forecasting":
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAPE", f"{mape:.2f}%")
    c1, c2, c3 = st.columns(3)
    c1.metric("Signal", signal_info["signal"])
    c2.metric("Expected Return", f"{signal_info['expected_return_percent']:.2f}%")
    c3.metric("Next-Day Predicted Close", f"{next_day_prediction:.2f}")

    st.markdown("### Backtest (Holdout Validation)")
    if backtest["status"] == "ok":
        b1, b2, b3 = st.columns(3)
        b1.metric(f"Backtest MAE ({backtest['test_days']}d)", f"{backtest['mae']:.2f}")
        b2.metric("Backtest RMSE", f"{backtest['rmse']:.2f}")
        b3.metric("Backtest MAPE", f"{backtest['mape_percent']:.2f}%")
    else:
        st.info("Backtest unavailable for current range. Try selecting a wider date range.")

    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Plot</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section visualizes the forecasted stock price for {} using a time series plot from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))
    if model is not None:
        forecast_plot = plot_plotly(model, forecast)
        st.plotly_chart(forecast_plot, width="stretch")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Predicted Close"))
        if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower Band", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper Band", line=dict(dash="dot")))
        fig.update_layout(title_text=f"{selected_stock} Forecast", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, width="stretch")

   
    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Components</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section breaks down the forecast components, including trends and seasonality, for {} from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))
    if model is not None:
        components = model.plot_components(forecast)
        st.write(components)
    else:
        st.info("Forecast component decomposition is available in local mode. Disable backend API mode to view it.")


if selected_tab == "Comparison":
    if selected_stocks:
       
        stocks_data = []
        forecasted_data = []
        for stock in selected_stocks:
            stocks_data.append(load_data(stock, start_date, end_date))

        st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Comparison Plot</h2>".format(', '.join(selected_stocks)), unsafe_allow_html=True)
        st.write("This section visualizes the forecasted stock price for {} using a time series plot from {} to {}.".format(', '.join(selected_stocks), end_date, end_date + pd.Timedelta(days=period)))

        for i, data in enumerate(stocks_data):
            if data is not None and not data.empty and {"Date", "Close"}.issubset(data.columns):
                df_train = prepare_training_frame(data)
                if len(df_train) < 30:
                    st.warning(f"Skipped {selected_stocks[i]}: not enough clean rows.")
                    continue
                model = Prophet()
                model.fit(df_train)
                future = model.make_future_dataframe(periods=period)
                forecast = model.predict(future)
                forecast = forecast[forecast['ds'] >= end_date_datetime]
                st.markdown("<h3><span style='color: orange;'>{}</span> Forecast DataFrame</h3>".format(selected_stocks[i]), unsafe_allow_html=True)

                
                new_forecast = prepare_forecast_table(forecast)
                st.dataframe(new_forecast, width="stretch")

                forecasted_data.append(forecast)

        if forecasted_data:
            plot_multiple_data(forecasted_data, selected_stocks)
        else:
            st.warning("No comparison forecasts could be generated for selected stocks.")
    else:
        st.warning("Please select at least one stock if you want to compare them.")

if selected_tab == "Interview View":
    st.markdown("## Full-Stack Interview Highlights")
    st.markdown(
        """
        - **Data layer:** Cached Yahoo Finance adapter with schema normalization for stable pandas processing.
        - **Domain layer:** Forecast pipeline (`prepare -> train -> predict -> evaluate`) with validation gates.
        - **Presentation layer:** Tabbed dashboard, downloadable datasets, and KPI metrics.
        - **Reliability:** Graceful handling for missing columns, empty data, and invalid date ranges.
        - **Scalability path:** Move `load_data` and forecasting into REST endpoints and keep Streamlit as UI client.
        """
    )

    st.markdown("### Runtime Health Snapshot")
    st.json(
        {
            "ticker": selected_stock,
            "training_rows": int(len(df_train)),
            "forecast_days": int(period),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "mape_percent": round(float(mape), 4),
            "signal": signal_info["signal"],
            "expected_return_percent": round(float(signal_info["expected_return_percent"]), 4),
            "backtest": backtest,
            "status": "healthy",
        }
    )

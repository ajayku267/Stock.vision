import os
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date, datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Conditional imports for cloud deployment
try:
    from services import (
        load_data, plot_data, plot_multiple_data, plot_volume,
        generate_prediction_bundle, fetch_backend_prediction,
        get_real_time_stock_data, get_multiple_real_time_data, get_market_overview,
        get_financial_news, get_trending_stocks, get_market_sentiment,
        plot_real_time_chart, display_stock_metrics
    )
    from ui_components import (
        render_sidebar_config, render_header, render_kpi_cards,
        render_dataframes_tab, render_plots_tab, render_statistics_tab,
        render_forecasting_tab, render_comparison_tab, render_interview_view
    )
    ML_FEATURES_AVAILABLE = True
except ImportError as e:
    st.error(f"Some advanced features are unavailable in cloud deployment: {str(e)}")
    ML_FEATURES_AVAILABLE = False

# Basic services that should always work
try:
    from services import load_data, plot_data, plot_volume
    from ui_components import render_sidebar_config, render_header, render_kpi_cards, render_dataframes_tab, render_plots_tab, render_statistics_tab
except ImportError:
    st.error("Core services not available")
    st.stop()


# Render UI components
render_header()
config = render_sidebar_config()

# Extract config variables
start_date = config["start_date"]
end_date = config["end_date"]
selected_stock = config["selected_stock"]
selected_stocks = config["selected_stocks"]
years_to_predict = config["years_to_predict"]
backtest_days = config["backtest_days"]
signal_threshold_pct = config["signal_threshold_pct"]
use_backend_api = config["use_backend_api"]
backend_url = config["backend_url"]
period = config["period"]

# Define available tabs based on feature availability
if ML_FEATURES_AVAILABLE:
    tab_options = ["Real-Time Data", "Market Overview", "Financial News", "Trending Stocks", "Dataframes", "Plots", "Statistics", "Forecasting", "Comparison", "Interview View"]
    tab_icons = ["activity", "graph-up", "newspaper", "trending-up", "table", "bar-chart", "calculator", "graph-up-arrow", "arrow-down-up", "briefcase"]
    default_tab = 0  # Start with Real-Time Data
else:
    tab_options = ["Dataframes", "Plots", "Statistics"]
    tab_icons = ["table", "bar-chart", "calculator"]
    default_tab = 0  # Start with Dataframes

selected_tab = option_menu(
    menu_title=None,
    options=tab_options,
    icons=tab_icons,
    menu_icon="📊",
    default_index=default_tab,
    orientation="horizontal",
)
# Load basic data for all modes
data = None
new_data = None
new_forecast = None
stats_summary_df = None
tech_data = None
df_train = None
forecast = None
model = None
mae = rmse = mape = 0.0
backtest = {}
signal_info = {"signal": "UNKNOWN", "expected_return_percent": 0.0}
next_day_prediction = 0.0

# Load basic stock data
with st.spinner("Loading stock data..."):
    try:
        data = load_data(selected_stock, start_date, end_date)
        if data is not None and not data.empty:
            new_data = data.copy()
            # Create basic stats
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            stats_summary_df = data[numeric_cols].describe()
            
            # Add basic technical indicators
            data["SMA_20"] = data["Close"].rolling(window=20).mean()
            data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
            delta = data["Close"].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            data["RSI_14"] = 100 - (100 / (1 + rs))
            data["Volatility_30"] = data["Close"].pct_change().rolling(window=30).std() * np.sqrt(252)
            tech_data = data.copy()
        else:
            st.error("Failed to load stock data")
            st.stop()
    except Exception as exc:
        st.error(f"Data loading failed: {str(exc)}")
        st.stop()

# Load ML features if available
if ML_FEATURES_AVAILABLE and (selected_tab in ["Forecasting", "Comparison", "Interview View"] or use_backend_api):
    with st.spinner("Loading ML features..."):
        try:
            if use_backend_api:
                api_payload = fetch_backend_prediction(
                    ticker=selected_stock,
                    config={
                        "backend_url": backend_url,
                        "start_date": start_date,
                        "end_date": end_date,
                        "years_to_predict": years_to_predict,
                        "backtest_days": backtest_days,
                        "signal_threshold_pct": signal_threshold_pct
                    }
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
            st.warning(f"ML features unavailable: {str(exc)}")
            ML_FEATURES_AVAILABLE = False

# Extract metrics for KPI display
metrics = {
    "training_rows": len(df_train),
    "mae": mae,
    "rmse": rmse,
    "mape": mape
}

render_kpi_cards(config, metrics, signal_info)

# Real-Time Data Tab
if selected_tab == "Real-Time Data":
    st.header("📈 Real-Time Stock Data")
    st.subheader(f"Current data for {selected_stock}")
    
    with st.spinner("Fetching real-time data..."):
        real_time_data = get_real_time_stock_data(selected_stock)
    
    if real_time_data:
        # Display metrics
        display_stock_metrics(real_time_data)
        
        # Display chart
        plot_real_time_chart(selected_stock, real_time_data)
        
        # Display additional information
        st.subheader("Company Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Company Name:**", real_time_data["company_name"])
            st.write("**Ticker:**", real_time_data["ticker"])
            st.write("**Last Updated:**", real_time_data["timestamp"])
        
        with col2:
            if real_time_data.get("revenue"):
                st.write("**Revenue:**", f"${real_time_data['revenue']:,.0f}")
            if real_time_data.get("eps"):
                st.write("**EPS:**", f"${real_time_data['eps']:.2f}")
            if real_time_data.get("beta"):
                st.write("**Beta:**", f"{real_time_data['beta']:.2f}")
    else:
        st.error("Failed to fetch real-time data. Please try again.")

# Market Overview Tab
if selected_tab == "Market Overview":
    st.header("🌍 Market Overview")
    
    with st.spinner("Fetching market overview..."):
        market_data = get_market_overview()
    
    if market_data and market_data.get("indices"):
        st.subheader("Market Indices")
        
        indices_data = market_data["indices"]
        for symbol, data in indices_data.items():
            if data and not isinstance(data, str):  # Check if data is valid
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        symbol,
                        f"${data.current_price:.2f}" if hasattr(data, 'current_price') else "N/A",
                        f"{data.change:+.2f}" if hasattr(data, 'change') else "N/A",
                        delta_color="normal" if (hasattr(data, 'change') and data.change >= 0) else "inverse"
                    )
                
                with col2:
                    if hasattr(data, 'volume') and data.volume:
                        st.metric("Volume", f"{data.volume:,}")
                
                with col3:
                    if hasattr(data, 'change_percent') and data.change_percent:
                        st.metric("Change %", f"{data.change_percent:+.2f}%")
                
                st.divider()
    
    st.subheader("Market Status")
    st.write(f"**Status:** {market_data.get('market_status', 'Unknown')}")
    st.write(f"**Last Updated:** {market_data.get('timestamp', 'Unknown')}")

# Financial News Tab
if selected_tab == "Financial News":
    st.header("📰 Financial News")
    
    # Option to filter by ticker
    news_filter = st.radio("News Filter", ["All News", f"{selected_stock} News"])
    
    with st.spinner("Fetching financial news..."):
        if news_filter == f"{selected_stock} News":
            news = get_financial_news(selected_stock, limit=15)
        else:
            news = get_financial_news(limit=20)
    
    if news:
        for article in news:
            with st.expander(f"**{article['title']}** - {article['source']}"):
                st.write(f"**Published:** {article['published_date']}")
                if article.get('author'):
                    st.write(f"**Author:** {article['author']}")
                st.write(f"**Summary:** {article['summary']}")
                if article.get('tickers'):
                    st.write(f"**Related Tickers:** {', '.join(article['tickers'])}")
                st.markdown(f"[Read more]({article['url']})")
    else:
        st.info("No news articles found.")

# Trending Stocks Tab
if selected_tab == "Trending Stocks":
    st.header("🔥 Trending Stocks")
    
    with st.spinner("Fetching trending stocks..."):
        trending = get_trending_stocks()
        market_sentiment = get_market_sentiment()
    
    if trending:
        st.subheader("Trending by News Mentions (Last 24 Hours)")
        
        # Create a dataframe for better display
        trending_df = pd.DataFrame(list(trending.items()), columns=['Ticker', 'Mentions'])
        trending_df = trending_df.sort_values('Mentions', ascending=False)
        
        # Display as a table
        st.dataframe(trending_df, use_container_width=True)
        
        # Create a bar chart
        fig = go.Figure(data=[
            go.Bar(x=trending_df['Ticker'], y=trending_df['Mentions'])
        ])
        fig.update_layout(title='Stock Mentions in News', xaxis_title='Ticker', yaxis_title='Mentions')
        st.plotly_chart(fig, width="stretch")
    
    if market_sentiment:
        st.subheader("Market Sentiment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_score = market_sentiment.get('sentiment_score', 0)
            st.metric("Sentiment Score", f"{sentiment_score:.3f}")
        
        with col2:
            sentiment_label = market_sentiment.get('sentiment_label', 'NEUTRAL')
            color = "normal" if sentiment_label == "BULLISH" else "inverse" if sentiment_label == "BEARISH" else "off"
            st.metric("Sentiment", sentiment_label, delta_color=color)
        
        with col3:
            article_count = market_sentiment.get('article_count', 0)
            st.metric("Articles Analyzed", article_count)
        
        if market_sentiment.get('top_tickers'):
            st.subheader("Top Tickers by Sentiment")
            top_tickers = market_sentiment['top_tickers']
            
            for ticker, data in list(top_tickers.items())[:5]:
                sentiment = data.get('sentiment', 0)
                mentions = data.get('mentions', 0)
                st.write(f"**{ticker}:** Sentiment {sentiment:.3f} ({mentions} mentions)")

if selected_tab == "Dataframes":
    render_dataframes_tab(
        selected_stock, start_date, end_date, new_data, new_forecast, period
    )

if selected_tab == "Plots":
    render_plots_tab(data)


if selected_tab == "Statistics":
    render_statistics_tab(selected_stock, stats_summary_df, tech_data)

if selected_tab == "Forecasting":
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
    render_forecasting_tab(
        selected_stock, end_date, period, metrics, signal_info, 
        next_day_prediction, backtest, forecast, model, df_train
    )


if selected_tab == "Comparison":
    render_comparison_tab(selected_stocks, start_date, end_date, period)

if selected_tab == "Interview View":
    metrics = {
        "training_rows": len(df_train),
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
    render_interview_view(
        selected_stock, config, metrics, signal_info, backtest
    )

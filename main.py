from time import sleep
import uuid
import pandas as pd
from sklearn.metrics import mean_absolute_error
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from services import load_data, plot_data, plot_multiple_data, plot_volume
st.set_page_config(layout="wide", page_title="StockVision", page_icon="📈")
st.sidebar.markdown("<h1 style='text-align: center; font-size: 30px;'><b>Stock.</b><b style='color: orange'>Vision</b></h1>", unsafe_allow_html=True)
st.sidebar.title("Options")
start_date_key = str(uuid.uuid4())
start_date = st.sidebar.date_input("Start date", date(2018, 1, 1), key=start_date_key)
end_date = st.sidebar.date_input("End date", date.today())
st.markdown("<h1 style='text-align: center;'>Stock Forecasting App 📈</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>Stock.</b><b style='color: orange'>Vision</b> is a simple web app for stock price prediction</p>", unsafe_allow_html=True)

selected_tab = option_menu(
    menu_title=None,
    options=["Dataframes", "Plots", "Statistics", "Forecasting", "Comparison"],
    icons=["table", "bar-chart", "calculator", "graph-up-arrow", "arrow-down-up"],
    menu_icon="📊",
    default_index=0,
    orientation="horizontal",
)
stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMC", "TSLA", "AMZN", "NFLX", "NVDA", "AMD", "PYPL")
selected_stock = st.sidebar.selectbox("Select stock for prediction", stocks)
selected_stocks = st.sidebar.multiselect("Select stocks for comparison", stocks)

years_to_predict = st.sidebar.slider("Years of prediction:", 1, 5)
period = years_to_predict * 365

with st.spinner("Loading data..."):
    data = load_data(selected_stock, start_date, end_date)
    sleep(1)

success_message = st.success("Data loaded successfully!")

sleep(1)

success_message.empty()

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

end_date_datetime = pd.to_datetime(end_date.strftime('%Y-%m-%d'))

forecast = forecast[forecast['ds'] >= end_date_datetime]

if selected_tab == "Dataframes":
    st.markdown("<h2><span style='color: orange;'>{}</span> Historical Data</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section displays historical stock price data for {} from {} to {}.".format(selected_stock, start_date, end_date))

    new_data = data.copy()

    new_data = data.drop(columns=['Adj Close', 'Volume'])
    st.dataframe(new_data, use_container_width=True)

    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Data</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section displays the forecasted stock price data for {} using the Prophet model from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))

    new_forecast = forecast.copy()

    new_forecast = new_forecast.drop(columns=[
        'additive_terms',
        'additive_terms_lower',
        'additive_terms_upper',
        'weekly',
        'weekly_lower',
        'weekly_upper',
        'yearly',
        'yearly_lower',
        'yearly_upper',
        'multiplicative_terms',
        'multiplicative_terms_lower',
        'multiplicative_terms_upper'
    ])

    new_forecast = new_forecast.rename(columns={
        "ds": "Date",
        "yhat": "Close",
        "yhat_lower": "Close Lower",
        "yhat_upper": "Close Upper",
        "trend": "Trend",
        "trend_lower": "Trend Lower",
        "trend_upper": "Trend Upper"
    })

    st.dataframe(new_forecast, use_container_width=True)

if selected_tab == "Plots":

    plot_data(data)

    plot_volume(data)


if selected_tab == "Statistics":
    st.markdown("<h2><span style='color: orange;'>Descriptive </span>Statistics</h2>", unsafe_allow_html=True)
    st.write("This section provides descriptive statistics for the selected stock.")

 
    data = data.drop(columns=['Date', 'Adj Close', 'Volume'])
    st.table(data.describe())

if selected_tab == "Forecasting":

    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Plot</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section visualizes the forecasted stock price for {} using a time series plot from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))
    forecast_plot = plot_plotly(model, forecast)
    st.plotly_chart(forecast_plot, use_container_width=True)

   
    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Components</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section breaks down the forecast components, including trends and seasonality, for {} from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))
    components = model.plot_components(forecast)
    st.write(components)


if selected_tab == "Comparison":
    if selected_stocks:
       
        stocks_data = []
        forcasted_data = []
        for stock in selected_stocks:
            stocks_data.append(load_data(stock, start_date, end_date))

        st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Comparison Plot</h2>".format(', '.join(selected_stocks)), unsafe_allow_html=True)
        st.write("This section visualizes the forecasted stock price for {} using a time series plot from {} to {}.".format(', '.join(selected_stocks), end_date, end_date + pd.Timedelta(days=period)))

        for i, data in enumerate(stocks_data):
            if data is not None:
                df_train = data[["Date", "Close"]]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
                model = Prophet()
                model.fit(df_train)
                future = model.make_future_dataframe(periods=period)
                forecast = model.predict(future)
                forecast = forecast[forecast['ds'] >= end_date_datetime]
                st.markdown("<h3><span style='color: orange;'>{}</span> Forecast DataFrame</h3>".format(selected_stocks[i]), unsafe_allow_html=True)

                
                new_forecast = forecast.copy()

           
                new_forecast = new_forecast.drop(columns=[
                    'additive_terms',
                    'additive_terms_lower',
                    'additive_terms_upper',
                    'weekly',
                    'weekly_lower',
                    'weekly_upper',
                    'yearly',
                    'yearly_lower',
                    'yearly_upper',
                    'multiplicative_terms',
                    'multiplicative_terms_lower',
                    'multiplicative_terms_upper'
                ])

                
                new_forecast = new_forecast.rename(columns={
                    "ds": "Date",
                    "yhat": "Close",
                    "yhat_lower": "Close Lower",
                    "yhat_upper": "Close Upper",
                    "trend": "Trend",
                    "trend_lower": "Trend Lower",
                    "trend_upper": "Trend Upper"
                })

                st.dataframe(new_forecast, use_container_width=True)

                forcasted_data.append(forecast)

        plot_multiple_data(forcasted_data, selected_stocks)
    else:
        st.warning("Please select at least one stock if you want to compare them.")

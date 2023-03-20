import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("StockUp")

stocks = ['ADANIPOWER.NS','ADANIPORTS.NS', 'APOLLOHOSP.NS',
'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS',
'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS',
'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
'HINDALCO.NS', 'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS',
'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK', 'LT.NS',
'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS',
'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS',
'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS',
'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']

selected_stocks = st.selectbox("Select dataset for predictions", stocks)

n_years = st.slider("Year of Predictions:", 1, 4)

period = n_years * 365


# Function to get the data from the Yahoo finance API
# Ticker is Basically is Stock Name
# The stock that we will select will be loaded in CACHE so we do  not have to load data
# again and again
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # This will place the date in the very first column
    return data


data_load_state = st.text(" Load Data... ")
data = load_data(selected_stocks)
data_load_state.text(" Loading data ... done!")

# now we will analyse the data that we have imported

st.subheader("Raw Data of the Stock")
# we can directly have the data plotting through streamlit
st.write(data.tail())


# Function to plot data more attractively
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# forecasting of stock
df_train = data[['Date','Close']]
# we have declared dictionary because facebook prophet needs data in this name only
# Above is referred from the Prophet Documentation
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# creating the facebook prophet model
# Prophet is similar to sklearn API
m = Prophet()

# now will fit the data in the model
m.fit(df_train)

# for forecasting we need to have the future frame

future = m.make_future_dataframe(periods=period)

# now storing the forecasted data

forecast = m.predict(future)

st.subheader("Forecasted Data")
# we can directly have the data plotting through streamlit
st.write(forecast.tail())

# plotting data more attractively

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)

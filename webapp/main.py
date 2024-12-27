import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os

TF_ENABLE_ONEDNN_OPTS = 0

@st.cache_resource
def load_model_and_scaler(stock):
    """load the model and scaler"""
    model = tf.keras.models.load_model(f'models/{stock}_model.keras')
    scaler = joblib.load(f'models/scaler_{stock}.pkl')
    print("+++++++++++++++++++++", os.path.basename(f'models/scaler_{stock}.pkl'))
    return model, scaler

def prepare_data_for_lstm(data, scaler, time_step=60):
    """prepare the data (reshaping for LSTM)"""
    data_scaled = scaler.transform(data['Close'].values.reshape(-1, 1))
    x_data, y_data = [], []

    for i in range(time_step, len(data_scaled)):
        x_data.append(data_scaled[i-time_step:i, 0])
        y_data.append(data_scaled[i, 0])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)

    return x_data, y_data, data_scaled


def make_predictions(data, model, scaler, days=7):
    """prediction"""
    data = np.array(data['Close']).reshape(-1, 1)
    data_scaled = scaler.transform(data)

    predictions = []

    last_sequence = data_scaled[-1].reshape(1, 1, 1)
    for i in range(days):
        next_pred = model.predict(last_sequence)
        predictions.append(next_pred[0][0])
        last_sequence = np.append(last_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

def plot_data(actual, predicted):
    fig = px.line(title='Actual vs Predicted Prices')
    fig.add_scatter(x=actual['Date'], y=actual['Close'], mode='lines', name='Actual Data')

    if actual.empty:
        st.error("⚠ No actual data available to plot.")
        return

    last_date = actual['Date'].iloc[-1]

    future_dates = [last_date + timedelta(days=i+1) for i in range(len(predicted))]

    fig.add_scatter(x=future_dates, y=predicted.flatten(), mode='lines', name='Predictions')
    st.plotly_chart(fig)


st.title("Stock Price Prediction App")
st.write("Choose a stock and see the predicted closing price for the next 7 days!")

tickers_names = {
  "Apple Inc.": ("AAPL", "apple"),
  "Xiaomi Corporation": ("1810.HK", "xiaomi"),
  "NVIDIA Corporation": ("NVDA", "nvidia"),
  "Alphabet Inc.": ("GOOG", "google"),
  "Amazon.com Inc.": ("AMZN", "amazon"),
  "Samsung Electronics": ("005930.KS","samsung")
}
stock = st.selectbox("Select stock", tickers_names.keys())

model, scaler = load_model_and_scaler(tickers_names[stock][1])

today = date.today()
yesterday = today - timedelta(days=1)

if st.button("Predict"):
    df = yf.download(tickers_names[stock][0], start='2020-01-01', end=yesterday)
    df.columns = [c[0] for c in df.columns]
    df = df.reset_index()

    st.success("✔ Data downloaded successfully!")
    if df.empty:
        st.error("⚠ No data available for the selected date. Please try a different date.")
    else:
        predictions = make_predictions(df, model, scaler)
        plot_data(df, predictions)

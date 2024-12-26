import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
import plotly.graph_objects as go

# Function to load the model and scaler
@st.cache_resource
def load_model_and_scaler(stock):
  model = tf.keras.models.load_model(f'models/{stock}_model.keras')
  scaler = joblib.load(f'models/scaler_{stock}_model.pkl')
  return model, scaler

# Function to make predictions
def make_prediction(model, scaler, data, n_predictions):
    data_scaled = scaler.transform(data)
    data_scaled = np.array(data_scaled).reshape((1, data_scaled.shape[0], 1))
    predictions = []
    for _ in range(n_predictions):
        prediction = model.predict(data_scaled)
        prediction = scaler.inverse_transform(prediction)
        prediction = prediction.reshape(1, 1)  # Flatten prediction
        predictions.append(prediction[0][0])

        # Keep only the last timestep
        data_scaled = np.append(data_scaled[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
    return predictions


# Function to plot the data
def plot_data(df, predictions, stock, future_dates):
    fig = go.Figure()

    # Add actual historical closing prices
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Actual Prices',
        line=dict(color='blue')
    ))

    # Add predicted closing prices for future dates
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name='Predicted Prices',
        line=dict(color='green', dash='dot')
    ))

    fig.update_layout(
        title=f"LSTM Predictions for {stock}",
        xaxis_title="Date",
        yaxis_title="Closing Prices",
        height=500,
        margin=dict(t=50, b=50)
    )

    st.plotly_chart(fig)


# Streamlit app
st.title("Stock Price Prediction App")
st.write("Choose a stock and see the predicted closing price for the next 7 days!")

# Select stock
tickers_names = {
  "Apple Inc.": "AAPL",
  "Xiaomi Corporation": "1810.HK",
  "NVIDIA Corporation": "NVDA",
  "Alphabet Inc.": "GOOG",
  "Amazon.com Inc.": "AMZN",
  "Samsung Electronics": "005930.KS",
}
stock = st.selectbox("Select stock", tickers_names.keys())

# Load model and scaler
model, scaler = load_model_and_scaler("apple")

# Date inputs for prediction
today = date.today()
yesterday = today - timedelta(days=1)

# Make prediction
if st.button("Predict"):
    # Fetch data for the selected stock
    df = yf.download(tickers_names[stock], start='2020-01-01', end=yesterday)
    df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    st.success("✔ Data downloaded successfully!")
    if df.empty:
        st.error("⚠ No data available for the selected date. Please try a different date.")
    else:
        df = df.reset_index()
        data = df[['Close']].values
        n_predictions = 7  # Number of days to predict

        predictions = make_prediction(model, scaler, data, n_predictions)
        future_dates = [today + timedelta(days=i) for i in range(n_predictions)]

        st.write(f"Dataset for {stock}")
        st.dataframe(df)

        # Plot the actual data and predictions
        plot_data(df, predictions, stock, future_dates)

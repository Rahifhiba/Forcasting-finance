import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import yfinance as yf
import tensorflow as tf
import joblib

st.title("Forecasting: Stock Prediction")

# Load the model and scaler
model = tf.keras.models.load_model("models/apple_model.keras")
scaler_apple = joblib.load("models/scaler_apple_model.pkl")

# Define ticker names and their respective symbols
tickers_names = {
    "Apple Inc.": "AAPL",
    "Xiaomi Corporation": "1810.HK",
    "NVIDIA Corporation": "NVDA",
    "Alphabet Inc.": "GOOG",
    "Amazon.com Inc.": "AMZN",
    "Samsung Electronics": "005930.KS",
}

# User selects a ticker
selected_ticker_name = st.selectbox("Select a stock for prediction:", tickers_names.keys())
selected_ticker = tickers_names[selected_ticker_name]

# Download data using yfinance
try:
    st.write(f"Downloading data for {selected_ticker_name} ({selected_ticker})...")
    data = yf.download(selected_ticker, start="2020-01-01", end="2023-12-31")
    data.columns = [c[0] for c in data.columns]

    data = data.reset_index()
    st.success("✔ Data downloaded successfully!")
except Exception as e:
    st.error(f"Error downloading data: {e}")
    data = None

if data is not None:
    st.subheader(f"Uploaded Data {selected_ticker_name} Preview")
    st.write(data.head())
    st.write("Preprocessing data for forecasting...")
    scaled_data = scaler_apple.transform(data['Close'].values.reshape(-1, 1))

    # Get user-defined prediction horizon
    forecast_horizon = st.slider("Select prediction horizon (days):", 1, 30, 7)

    # Generate predictions
    predictions = []
    input_data = scaled_data[-1].reshape(1, -1)  # Ensure input_data has the right shape
    for _ in range(forecast_horizon):
        next_prediction = model.predict(input_data)
        predictions.append(next_prediction[0, 0])  # Collect prediction
        # Update input data with the new prediction
        input_data = np.append(input_data[:, 1:], next_prediction, axis=-1).reshape(1, -1)

    # Inverse transform predictions back to original scale
    predictions = scaler_apple.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Prepare forecast dataframe
    future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_horizon + 1)[1:]
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": predictions})

    # Plot forecast results
    st.subheader(f"{selected_ticker_name} Forecast for the Next {forecast_horizon} Days")
    fig = px.line(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        labels={"x": "Date", "y": "Price"},
        title=f"Forecast for {selected_ticker_name}",
    )
    st.plotly_chart(fig)

    # Show forecast dataframe
    st.write("Forecasted Prices:")
    st.write(forecast_df)

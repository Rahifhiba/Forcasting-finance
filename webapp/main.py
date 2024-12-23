#!/usr/bin/env python3

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import yfinance as yf
import tensorflow as tf

st.title("Forecasting: Stock prediction")
# model = tf.keras.models.load_model("")

st.write("Upload your stock data (CSV format) to predict future prices.")

uploaded_file = st.file_uploader("Choose a file")
st.write("or")
with open("company_tickers.json", "r") as file:
    symbols = json.load(file)
tickers = [data["ticker"] for data in symbols.values()]

selected_ticker = st.selectbox("Select dataset for prediction", tickers)

try:
    yf.download(selected_ticker, start="2020-01-01", end="2023-12-31")
except Exception as e:
    st.write("Error:", e)

if uploaded_file is not None or selected_ticker is not None:
    user_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.write(user_data.head())

    user_data["Close"] = scaler.transform(user_data[["Close"]])
    X_user, _ = create_features(user_data)

    # Predict
    # predictions = model.predict(X_user)
    # user_data["Predicted_Close"] = scaler.inverse_transform(predictions)

    # st.write("Predictions:")
    # st.write(user_data[["Date", "Close", "Predicted_Close"]])

    # # Plot
    # st.line_chart(user_data[["Close", "Predicted_Close"]])

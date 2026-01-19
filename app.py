# app.py

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import kagglehub
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Gold Price Forecasting",
    layout="wide"
)

st.title("📈 Gold Price Forecasting using ARIMA")
st.write("Data source: Kaggle (downloaded automatically using kagglehub)")

# ----------------------------
# Download dataset using kagglehub
# ----------------------------
with st.spinner("Downloading dataset from Kaggle using kagglehub..."):
    path = kagglehub.dataset_download(
        "limbaddd/gold-price-predictions"
    )

st.success("Dataset downloaded successfully!")
st.write("📁 Dataset path:", path)

# ----------------------------
# Load CSV file
# ----------------------------
csv_files = glob.glob(os.path.join(path, "*.csv"))

if not csv_files:
    st.error("No CSV file found in the dataset.")
    st.stop()

csv_file = csv_files[0]
df = pd.read_csv(csv_file)

st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Detect Date & Target columns
# ----------------------------
date_col = [c for c in df.columns if "date" in c.lower()][0]
price_col = [c for c in df.columns if "close" in c.lower() or "price" in c.lower()][0]

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)
df.set_index(date_col, inplace=True)

ts = df[price_col].copy()
ts = ts.ffill().bfill()

# ----------------------------
# Plot time series
# ----------------------------
st.subheader("Gold Price Time Series")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ts, label="Gold Price")
ax.set_title("Gold Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ----------------------------
# Stationarity Test
# ----------------------------
st.subheader("Stationarity Test (ADF)")

adf_stat, p_value, _, _, _, _ = adfuller(ts)
st.write(f"ADF Statistic: **{adf_stat:.4f}**")
st.write(f"p-value: **{p_value:.6f}**")

if p_value < 0.05:
    st.success("Series is stationary")
else:
    st.warning("Series is non-stationary (differencing required)")

# ----------------------------
# Train-Test Split
# ----------------------------
split = int(len(ts) * 0.8)
train = ts[:split]
test = ts[split:]

# Log transform
train_log = np.log(train)
test_log = np.log(test)

# ----------------------------
# ARIMA Model
# ----------------------------
st.subheader("ARIMA Model Fitting")

with st.spinner("Training ARIMA model..."):
    model = auto_arima(
        train_log,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True
    )

st.write("Selected ARIMA Model:")
st.write(model)

# ----------------------------
# Forecast on Test Data
# ----------------------------
forecast_log = model.predict(n_periods=len(test_log))
forecast = np.exp(forecast_log)

# ----------------------------
# Evaluation
# ----------------------------
mae = mean_absolute_error(test, forecast)
rmse = mean_squared_error(test, forecast, squared=False)

st.subheader("Model Evaluation")
st.metric("MAE", f"{mae:.4f}")
st.metric("RMSE", f"{rmse:.4f}")

# ----------------------------
# Plot Forecast vs Actual
# ----------------------------
st.subheader("Forecast vs Actual")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train, label="Train")
ax.plot(test, label="Test")
ax.plot(test.index, forecast, label="Forecast")
ax.set_title("ARIMA Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Gold Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ----------------------------
# Future Forecast
# ----------------------------
st.subheader("Future Gold Price Forecast")

future_days = st.slider("Select number of days to forecast", 7, 60, 30)

with st.spinner("Forecasting future prices..."):
    future_log = model.predict(n_periods=future_days)
    future_prices = np.exp(future_log)

future_dates = pd.date_range(
    start=ts.index.max() + pd.Timedelta(days=1),
    periods=future_days
)

future_df = pd.DataFrame(
    {"Forecasted Gold Price": future_prices},
    index=future_dates
)

st.line_chart(future_df)
st.dataframe(future_df.head())

st.success("Forecasting completed successfully ✅")

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
    path = kagglehub.dataset_download("limbaddd/gold-price-predictions")

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
# Detect Date & Target columns (safer)
# ----------------------------
date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
if not date_candidates:
    st.error("No Date column found. Please ensure the dataset has a Date/Time column.")
    st.stop()
date_col = date_candidates[0]

price_candidates = [c for c in df.columns if ("close" in c.lower()) or ("price" in c.lower())]
if not price_candidates:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("No numeric price column found for forecasting.")
        st.stop()
    price_col = num_cols[0]
else:
    price_col = price_candidates[0]

st.write(f"✅ Detected Date column: **{date_col}**")
st.write(f"✅ Detected Price column: **{price_col}**")

# Parse dates safely
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col)
df.set_index(date_col, inplace=True)

# ----------------------------
# Build time series + CLEANING (CRITICAL FIX)
# ----------------------------
ts = df[price_col].copy()

# Convert strings like "1,234.56" to numeric
if ts.dtype == "object":
    ts = ts.astype(str).str.replace(",", "", regex=False)

ts = pd.to_numeric(ts, errors="coerce")

# Remove invalid values
ts = ts.replace([np.inf, -np.inf], np.nan)
ts = ts.dropna()

# Remove duplicate timestamps if any
ts = ts[~ts.index.duplicated(keep="last")]

# Fill any remaining gaps (optional safety)
ts = ts.ffill().bfill()

# Final safety check
if ts.empty or len(ts) < 30:
    st.error("Time series is empty or too short after cleaning. Cannot proceed.")
    st.stop()

st.write(f"📌 Final cleaned series length: **{len(ts)}**")
st.write(f"📌 Series dtype: **{ts.dtype}**")

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
# Stationarity Test (FIXED)
# ----------------------------
st.subheader("Stationarity Test (ADF)")

try:
    adf_result = adfuller(ts.dropna())
    adf_stat = adf_result[0]
    p_value = adf_result[1]

    st.write(f"ADF Statistic: **{adf_stat:.4f}**")
    st.write(f"p-value: **{p_value:.6f}**")

    if p_value < 0.05:
        st.success("Series is stationary (reject unit root at 5% level)")
    else:
        st.warning("Series is non-stationary (differencing required)")
except Exception as e:
    st.error("ADF test failed due to invalid series values.")
    st.write("Error details:", str(e))
    st.stop()

# ----------------------------
# Train-Test Split
# ----------------------------
split = int(len(ts) * 0.8)
train = ts.iloc[:split]
test = ts.iloc[split:]

if len(train) < 20 or len(test) < 5:
    st.error("Not enough data points after split to train/test the model.")
    st.stop()

# ----------------------------
# Safe log transform (FIXED)
# ----------------------------
min_val = min(train.min(), test.min())
shift = 0.0
if min_val <= 0:
    shift = abs(min_val) + 1e-6
    st.warning(f"⚠️ Non-positive values detected. Applying shift of {shift:.6f} before log transform.")

train_log = np.log(train + shift)
test_log = np.log(test + shift)

# ----------------------------
# ARIMA Model
# ----------------------------
st.subheader("ARIMA Model Fitting")

with st.spinner("Training ARIMA model..."):
    model = auto_arima(
        train_log,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )

st.write("Selected ARIMA Model:")
st.write(model)

# ----------------------------
# Forecast on Test Data
# ----------------------------
forecast_log = model.predict(n_periods=len(test_log))
forecast = np.exp(forecast_log) - shift
forecast = pd.Series(forecast, index=test.index)

# ----------------------------
# ✅ Evaluation (FIXED: remove NaN/inf safely)
# ----------------------------
# Make both series finite
test_clean = test.replace([np.inf, -np.inf], np.nan)
forecast_clean = forecast.replace([np.inf, -np.inf], np.nan)

# Combine and drop NaNs from either side
eval_df = pd.concat(
    [test_clean.rename("Actual"), forecast_clean.rename("Forecast")],
    axis=1
).dropna()

if eval_df.empty:
    st.error("Forecast contains invalid values (NaN/inf). Model may be unstable for this dataset.")
    st.stop()

mae = mean_absolute_error(eval_df["Actual"], eval_df["Forecast"])
rmse = mean_squared_error(eval_df["Actual"], eval_df["Forecast"], squared=False)

st.subheader("Model Evaluation")
st.metric("MAE", f"{mae:.4f}")
st.metric("RMSE", f"{rmse:.4f}")

# ----------------------------
# Plot Forecast vs Actual (use cleaned forecast for plotting)
# ----------------------------
st.subheader("Forecast vs Actual")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train, label="Train")
ax.plot(test, label="Test")
ax.plot(forecast_clean.index, forecast_clean.values, label="Forecast")
ax.set_title("ARIMA Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Gold Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ----------------------------
# Future Forecast (also make finite)
# ----------------------------
st.subheader("Future Gold Price Forecast")

future_days = st.slider("Select number of days to forecast", 7, 60, 30)

with st.spinner("Forecasting future prices..."):
    future_log = model.predict(n_periods=future_days)
    future_prices = np.exp(future_log) - shift

future_prices = pd.Series(future_prices).replace([np.inf, -np.inf], np.nan).ffill().bfill().values

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

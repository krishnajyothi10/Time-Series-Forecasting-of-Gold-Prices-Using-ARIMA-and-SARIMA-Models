# app.py  (STABLE VERSION)

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


st.set_page_config(page_title="Gold Price Forecasting", layout="wide")
st.title("📈 Gold Price Forecasting using ARIMA")
st.write("Data source: Kaggle (downloaded automatically using kagglehub)")

# ----------------------------
# Download dataset
# ----------------------------
with st.spinner("Downloading dataset from Kaggle using kagglehub..."):
    path = kagglehub.dataset_download("limbaddd/gold-price-predictions")

st.success("Dataset downloaded successfully!")
st.write("📁 Dataset path:", path)

# ----------------------------
# Load CSV
# ----------------------------
csv_files = glob.glob(os.path.join(path, "*.csv"))
if not csv_files:
    st.error("No CSV file found in the dataset.")
    st.stop()

df = pd.read_csv(csv_files[0])
st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Detect columns
# ----------------------------
date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
if not date_candidates:
    st.error("No Date column found.")
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

# ----------------------------
# Parse dates + clean series
# ----------------------------
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

ts = df[price_col].copy()
if ts.dtype == "object":
    ts = ts.astype(str).str.replace(",", "", regex=False)
ts = pd.to_numeric(ts, errors="coerce")

ts = ts.replace([np.inf, -np.inf], np.nan).dropna()
ts = ts[~ts.index.duplicated(keep="last")]
ts = ts.ffill().bfill()

if ts.empty or len(ts) < 50:
    st.error("Time series is empty or too short after cleaning.")
    st.stop()

st.write(f"📌 Cleaned series length: **{len(ts)}**")

# ----------------------------
# Plot series
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
# ADF test (safe)
# ----------------------------
st.subheader("Stationarity Test (ADF)")
try:
    adf_stat, p_value, *_ = adfuller(ts)
    st.write(f"ADF Statistic: **{adf_stat:.4f}**")
    st.write(f"p-value: **{p_value:.6f}**")
    if p_value < 0.05:
        st.success("Series is stationary (reject unit root at 5% level)")
    else:
        st.warning("Series is non-stationary (differencing required)")
except Exception as e:
    st.warning("ADF test could not be computed for this series.")
    st.write(str(e))

# ----------------------------
# Train-test split
# ----------------------------
split = int(len(ts) * 0.8)
train = ts.iloc[:split]
test = ts.iloc[split:]

if len(train) < 30 or len(test) < 10:
    st.error("Not enough data after split.")
    st.stop()

# ----------------------------
# STABLE transform: log1p (requires non-negative)
# If any negative values exist, shift once.
# ----------------------------
min_val = min(train.min(), test.min())
shift = 0.0
if min_val < 0:
    shift = abs(min_val)

train_t = np.log1p(train + shift)
test_t  = np.log1p(test + shift)

# ----------------------------
# Model fit (more stable constraints)
# ----------------------------
st.subheader("ARIMA Model Fitting")

with st.spinner("Training ARIMA model..."):
    model = auto_arima(
        train_t,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=5, max_q=5,   # limit complexity for stability
        max_order=10
    )

st.write("Selected ARIMA Model:")
st.write(model)

# ----------------------------
# Forecast on test (STABLE inverse transform)
# Add clipping to avoid overflow
# ----------------------------
pred_t = model.predict(n_periods=len(test_t))
pred_t = np.asarray(pred_t)

# Clip transformed preds to safe range
pred_t = np.clip(pred_t, -20, 20)

forecast = np.expm1(pred_t) - shift
forecast = pd.Series(forecast, index=test.index)

# Clean forecast for metrics
forecast = forecast.replace([np.inf, -np.inf], np.nan)
test_clean = test.replace([np.inf, -np.inf], np.nan)

eval_df = pd.concat([test_clean.rename("Actual"), forecast.rename("Forecast")], axis=1).dropna()

if eval_df.empty:
    st.error("Model produced invalid forecasts again. Try reducing forecast horizon or using seasonal=False only.")
    st.stop()

# ----------------------------
# Metrics
# ----------------------------
mae = mean_absolute_error(eval_df["Actual"], eval_df["Forecast"])
rmse = mean_squared_error(eval_df["Actual"], eval_df["Forecast"], squared=False)

st.subheader("Model Evaluation")
c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.4f}")
c2.metric("RMSE", f"{rmse:.4f}")

# ----------------------------
# Plot forecast
# ----------------------------
st.subheader("Forecast vs Actual")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train, label="Train")
ax.plot(test, label="Test")
ax.plot(forecast.index, forecast.values, label="Forecast")
ax.set_title("ARIMA Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Gold Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ----------------------------
# Future forecast (stable)
# ----------------------------
st.subheader("Future Gold Price Forecast")
future_days = st.slider("Select number of days to forecast", 7, 60, 30)

future_t = model.predict(n_periods=future_days)
future_t = np.clip(np.asarray(future_t), -20, 20)
future_prices = np.expm1(future_t) - shift
future_prices = pd.Series(future_prices).replace([np.inf, -np.inf], np.nan).ffill().bfill().values

future_dates = pd.date_range(start=ts.index.max() + pd.Timedelta(days=1), periods=future_days)
future_df = pd.DataFrame({"Forecasted Gold Price": future_prices}, index=future_dates)

st.line_chart(future_df)
st.dataframe(future_df.head())

st.success("Forecasting completed successfully ✅")

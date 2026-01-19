# app.py (ROBUST ALWAYS-RUNS VERSION)

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
st.title("📈 Gold Price Forecasting (ARIMA) - KaggleHub")
st.write("Data source: Kaggle (downloaded automatically using kagglehub)")

# ----------------------------
# Download dataset
# ----------------------------
with st.spinner("Downloading dataset from Kaggle using kagglehub..."):
    path = kagglehub.dataset_download("limbaddd/gold-price-predictions")

st.success("Dataset downloaded successfully!")
st.write("📁 Dataset path:", path)

# ----------------------------
# Load CSV (allow selection if multiple)
# ----------------------------
csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))
if not csv_files:
    st.error("No CSV file found in the dataset.")
    st.stop()

csv_file = st.selectbox("Select CSV file", csv_files, index=0)

try:
    df = pd.read_csv(csv_file)
except UnicodeDecodeError:
    df = pd.read_csv(csv_file, encoding="latin1")

st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Detect date column (allow selection)
# ----------------------------
date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
if not date_candidates:
    # allow user to pick any column
    date_candidates = df.columns.tolist()

date_col = st.selectbox("Select Date/Time column", date_candidates, index=0)

# Parse dates safely
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

# ----------------------------
# Select target column (numeric only) - KEY FIX
# ----------------------------
# Convert object columns with commas to numeric candidates
df_clean = df.copy()
for c in df_clean.columns:
    if df_clean[c].dtype == "object":
        df_clean[c] = df_clean[c].astype(str).str.replace(",", "", regex=False)
    df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if df_clean[c].notna().sum() > 50]  # ensure enough data

if not numeric_cols:
    st.error("No usable numeric column found after cleaning. Please check dataset format.")
    st.stop()

default_idx = 0
for i, c in enumerate(numeric_cols):
    if "close" in c.lower():
        default_idx = i
        break

target_col = st.selectbox("Select Target (Price) Column", numeric_cols, index=default_idx)

ts = df_clean[target_col].copy()
ts = ts.replace([np.inf, -np.inf], np.nan).dropna()
ts = ts[~ts.index.duplicated(keep="last")]
ts = ts.ffill().bfill()

if ts.empty or len(ts) < 60:
    st.error("Time series is empty or too short after cleaning. Choose another target column.")
    st.stop()

st.write(f"✅ Using target column: **{target_col}**")
st.write(f"📌 Series length: **{len(ts)}** | Date range: **{ts.index.min().date()}** to **{ts.index.max().date()}**")

# ----------------------------
# Plot series
# ----------------------------
st.subheader("Gold Price Time Series")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ts, label=target_col)
ax.set_title("Gold Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ----------------------------
# ADF test (optional)
# ----------------------------
st.subheader("Stationarity Test (ADF)")
try:
    adf_stat, p_value, *_ = adfuller(ts)
    st.write(f"ADF Statistic: **{adf_stat:.4f}**")
    st.write(f"p-value: **{p_value:.6f}**")
    if p_value < 0.05:
        st.success("Series is stationary (reject unit root at 5% level)")
    else:
        st.warning("Series is non-stationary (ARIMA will handle differencing)")
except Exception as e:
    st.warning("ADF test failed (not critical for forecasting).")
    st.write(str(e))

# ----------------------------
# Settings
# ----------------------------
with st.sidebar:
    st.header("Model Settings")
    test_pct = st.slider("Test size (%)", 10, 40, 20, 5)
    future_days = st.slider("Future forecast days", 7, 60, 30)
    max_p = st.slider("max_p", 1, 8, 5)
    max_q = st.slider("max_q", 1, 8, 5)
    max_d = st.slider("max_d", 0, 2, 2)

# ----------------------------
# Train-test split
# ----------------------------
split = int(len(ts) * (1 - test_pct / 100))
train = ts.iloc[:split]
test = ts.iloc[split:]

if len(train) < 40 or len(test) < 10:
    st.error("Not enough data after split. Reduce test size.")
    st.stop()

# ----------------------------
# Fit Auto-ARIMA DIRECTLY on original scale (NO LOG) - KEY FIX
# ----------------------------
st.subheader("ARIMA Model Fitting")

with st.spinner("Training ARIMA model (Auto-ARIMA)..."):
    model = auto_arima(
    train,
    seasonal=False,
    d=1,              # ✅ FORCE first differencing
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_p=5,
    max_q=5,
    information_criterion="aic"
)


st.write("Selected ARIMA Model:")
st.write(model)

# ----------------------------
# Predict on test
# ----------------------------
pred = model.predict(n_periods=len(test))
forecast = pd.Series(pred, index=test.index)

# Clean for evaluation
forecast = forecast.replace([np.inf, -np.inf], np.nan)
test_clean = test.replace([np.inf, -np.inf], np.nan)

eval_df = pd.concat([test_clean.rename("Actual"), forecast.rename("Forecast")], axis=1).dropna()

if eval_df.empty:
    st.error("Forecast still contains invalid values. Try a different target column or lower max_p/max_q.")
    st.stop()

# Metrics
mae = mean_absolute_error(eval_df["Actual"], eval_df["Forecast"])
rmse = mean_squared_error(eval_df["Actual"], eval_df["Forecast"], squared=False)

st.subheader("Model Evaluation")
c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.4f}")
c2.metric("RMSE", f"{rmse:.4f}")

# Plot forecast vs actual
st.subheader("Forecast vs Actual")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train, label="Train")
ax.plot(test, label="Test")
ax.plot(forecast, label="Forecast")
ax.set_title("ARIMA Forecast vs Actual")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ----------------------------
# Future forecast
# ----------------------------
st.subheader("Future Forecast")
future_pred = model.predict(n_periods=future_days)
future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(days=1), periods=future_days, freq="D")
future_series = pd.Series(future_pred, index=future_index).replace([np.inf, -np.inf], np.nan).ffill().bfill()

future_df = pd.DataFrame({"Forecasted Price": future_series})
st.line_chart(future_df)
st.dataframe(future_df.head(10))

st.success("Forecasting completed successfully ✅")


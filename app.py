# app.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima import auto_arima


# ----------------------------
# Helpers
# ----------------------------
def detect_date_column(df: pd.DataFrame) -> str:
    # Prefer columns containing 'date' or 'time'
    candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    return candidates[0] if candidates else df.columns[0]

def detect_target_column(df: pd.DataFrame) -> str:
    # Prefer close/price-like numeric columns
    preferred = ["close", "adj close", "adj_close", "price", "gold", "value"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for p in preferred:
        for c in df.columns:
            if p in c.lower() and c in numeric_cols:
                return c
    # fallback: numeric column with most non-null values
    if not numeric_cols:
        raise ValueError("No numeric columns found to forecast.")
    return df[numeric_cols].count().sort_values(ascending=False).index[0]

def adf_summary(series: pd.Series) -> dict:
    s = series.dropna()
    stat, pval, _, _, crit, _ = adfuller(s, autolag="AIC")
    return {"ADF Statistic": stat, "p-value": pval, "Critical Values": crit}

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def plot_series(x: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x.index, x.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    st.pyplot(fig)

def plot_train_test_forecast(train_actual, test_actual, forecast, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_actual.index, train_actual.values, label="Train")
    ax.plot(test_actual.index, test_actual.values, label="Test")
    ax.plot(forecast.index, forecast.values, label="Forecast")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Gold Price Forecasting (ARIMA)", layout="wide")
st.title("📈 Gold Price Forecasting (Time Series) — Simple Streamlit App")

st.write(
    "Upload a gold price CSV. The app will auto-detect the Date column and a numeric target "
    "(preferably Close/Price), run EDA, fit Auto-ARIMA, evaluate, and forecast future values."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

with st.sidebar:
    st.header("Settings")
    test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100
    forecast_horizon = st.number_input("Future forecast horizon (days)", min_value=7, max_value=365, value=30, step=1)
    enforce_daily = st.checkbox("Enforce daily frequency (fill gaps)", value=False)
    seasonal = st.checkbox("Try seasonal ARIMA (SARIMA)", value=False)
    m_season = st.selectbox("Season length m (if SARIMA)", options=[7, 12, 30, 365], index=0)

if uploaded is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# Load
try:
    df = pd.read_csv(uploaded)
except UnicodeDecodeError:
    df = pd.read_csv(uploaded, encoding="latin1")

st.subheader("1) Raw Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# Detect columns
date_col = detect_date_column(df)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

target_col = detect_target_column(df)
ts = df[target_col].copy()

# Clean
ts = ts.ffill().bfill()
ts = ts[~ts.index.duplicated(keep="last")]

if enforce_daily:
    ts = ts.asfreq("D")
    ts = ts.ffill().bfill()

st.subheader("2) Detected Columns")
c1, c2, c3 = st.columns(3)
c1.metric("Date column", date_col)
c2.metric("Target column", target_col)
c3.metric("Observations", f"{len(ts)}")

st.write(f"Date range: **{ts.index.min().date()}** to **{ts.index.max().date()}**")

# EDA
st.subheader("3) EDA")
plot_series(ts, f"Time Series Plot: {target_col}", target_col)

roll = ts.rolling(30).mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ts.index, ts.values, label="Original")
ax.plot(roll.index, roll.values, label="Rolling Mean (30)")
ax.set_title("Rolling Mean (Window=30)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ADF
st.subheader("4) Stationarity Check (ADF Test)")
adf_res = adf_summary(ts)
st.write(f"**ADF Statistic:** {adf_res['ADF Statistic']:.4f}")
st.write(f"**p-value:** {adf_res['p-value']:.6f}")
st.write("**Critical Values:**", adf_res["Critical Values"])
if adf_res["p-value"] < 0.05:
    st.success("Series appears stationary (reject unit root at 5%).")
else:
    st.warning("Series appears non-stationary (fail to reject unit root at 5%). ARIMA will handle differencing (d).")

# Train-test (log transform for stability)
st.subheader("5) Train–Test Split, Model Fit, and Evaluation")

ts_log = np.log(ts)
split = int(len(ts_log) * (1 - test_size))
train = ts_log.iloc[:split]
test = ts_log.iloc[split:]

st.write(f"Train size: **{len(train)}** | Test size: **{len(test)}**")

# ACF plot (on differenced train)
fig = plt.figure(figsize=(10, 3))
plot_acf(train.diff().dropna(), lags=40)
plt.title("ACF (Train, 1st Difference)")
st.pyplot(fig)

# Fit model
with st.spinner("Fitting model (auto_arima)..."):
    model = auto_arima(
        train,
        seasonal=seasonal,
        m=int(m_season) if seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )

st.write("**Selected model:**", model)

# Forecast on test
forecast_log = pd.Series(model.predict(n_periods=len(test)), index=test.index)
forecast = np.exp(forecast_log)
actual = np.exp(test)
train_actual = np.exp(train)

# Metrics
mae = mean_absolute_error(actual, forecast)
rmse = mean_squared_error(actual, forecast, squared=False)
mape_val = mape(actual, forecast)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.4f}")
m2.metric("RMSE", f"{rmse:.4f}")
m3.metric("MAPE (%)", f"{mape_val:.2f}")

plot_train_test_forecast(train_actual, actual, forecast, "Forecast vs Actual (Test Set)")

# Future forecast
st.subheader("6) Future Forecast")
with st.spinner("Training final model on full data and forecasting..."):
    final_model = auto_arima(
        ts_log,
        seasonal=seasonal,
        m=int(m_season) if seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    future_log = final_model.predict(n_periods=int(forecast_horizon))

# Create future index (daily)
future_index = pd.date_range(start=ts.index.max() + pd.Timedelta(days=1), periods=int(forecast_horizon), freq="D")
future_forecast = pd.Series(np.exp(future_log), index=future_index, name="Forecast")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ts.index, ts.values, label="Historical")
ax.plot(future_forecast.index, future_forecast.values, label="Future Forecast")
ax.set_title(f"Next {forecast_horizon} Days Forecast")
ax.set_xlabel("Date")
ax.set_ylabel(target_col)
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.write("Forecast table (first 10 rows):")
st.dataframe(future_forecast.head(10).to_frame(), use_container_width=True)

st.success("Done. You can change settings in the sidebar and re-run automatically.")

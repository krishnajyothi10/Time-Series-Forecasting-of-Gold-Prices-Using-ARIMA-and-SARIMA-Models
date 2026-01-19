# ----------------------------
# ARIMA Model Fitting (STABLE METHOD: fit on differenced series)
# ----------------------------
st.subheader("ARIMA Model Fitting (Stable - Differenced Series)")

# 1st difference to make series approximately stationary
train_diff = train.diff().dropna()

if len(train_diff) < 30:
    st.error("Not enough data after differencing to fit the model.")
    st.stop()

with st.spinner("Training ARMA model on differenced series (Auto-ARIMA)..."):
    # d=0 because we already differenced manually
    model = auto_arima(
        train_diff,
        seasonal=False,
        d=0,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=max_p,
        max_q=max_q,
        max_order=10,
        information_criterion="aic"
    )

st.write("Selected Model (on differenced series):")
st.write(model)

# ----------------------------
# Predict on test: forecast differences -> convert back to price levels
# ----------------------------
pred_diff = model.predict(n_periods=len(test))
pred_diff = pd.Series(pred_diff, index=test.index)

# Clean predicted diffs
pred_diff = pred_diff.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Convert diff forecast to level forecast
last_train_value = float(train.iloc[-1])
forecast = last_train_value + pred_diff.cumsum()

# Final cleaning
forecast = forecast.replace([np.inf, -np.inf], np.nan)
test_clean = test.replace([np.inf, -np.inf], np.nan)

eval_df = pd.concat(
    [test_clean.rename("Actual"), forecast.rename("Forecast")],
    axis=1
).dropna()

if eval_df.empty:
    st.error("Forecast still contains invalid values. Try a different target column.")
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
ax.set_title("ARIMA Forecast vs Actual (Stable Differenced Method)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ----------------------------
# Future Forecast (differences -> levels)
# ----------------------------
st.subheader("Future Forecast")

future_diff = model.predict(n_periods=future_days)
future_diff = pd.Series(future_diff).replace([np.inf, -np.inf], np.nan).fillna(0.0)

future_index = pd.date_range(
    start=ts.index.max() + pd.Timedelta(days=1),
    periods=future_days,
    freq="D"
)

last_value = float(ts.iloc[-1])
future_levels = last_value + future_diff.cumsum().values

future_df = pd.DataFrame({"Forecasted Price": future_levels}, index=future_index)
st.line_chart(future_df)
st.dataframe(future_df.head(10))

st.success("Forecasting completed successfully ✅")

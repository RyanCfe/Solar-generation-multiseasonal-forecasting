"""ATSA solar generation forecasting: Ryan (S041), Yugraj (S043), Aleric (S050)."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

figure_dir = Path("figures")
figure_dir.mkdir(exist_ok=True)
_figure_no = 0
def save_current_figure():
    global _figure_no
    _figure_no += 1
    plt.savefig(figure_dir / f"figure_{_figure_no:02d}.png", bbox_inches="tight", dpi=160)
    plt.close("all")


# Data preparation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("time_series_60min_singleindex_filtered.csv", usecols=["utc_timestamp", "DE_solar_generation_actual"])

# Keep only Germany hourly solar generation
solar = df[["utc_timestamp", "DE_solar_generation_actual"]].copy()

# Prophet uses ds for timestamp and y for target
solar = solar.rename(columns={
    "utc_timestamp": "ds",
    "DE_solar_generation_actual": "y"
})

# Convert text timestamp to datetime
solar["ds"] = pd.to_datetime(solar["ds"], utc=True)
solar.head()


# Data checks

print("Start:", solar["ds"].min())
print("End:", solar["ds"].max())

print("\nMissing values:")
print(solar.isna().sum())
print("Missing %:", round(solar["y"].isna().mean() * 100, 3))

print("\nDuplicate timestamps:", solar["ds"].duplicated().sum())

print("\nMost common time gaps:")
print(solar["ds"].diff().value_counts().head())

print("\nGeneration statistics:")
print(solar["y"].describe())

# Time series plots

# Full series
plt.figure(figsize=(15, 5))
plt.plot(solar["ds"], solar["y"])
plt.title("Germany Hourly Solar Power Generation")
plt.xlabel("Time")
plt.ylabel("Solar Generation (MW)")
save_current_figure()

# One-week zoom
week = solar[(solar["ds"] >= "2019-06-03") & (solar["ds"] < "2019-06-10")]

plt.figure(figsize=(15, 5))
plt.plot(week["ds"], week["y"])
plt.title("Germany Solar Generation — One Week")
plt.xlabel("Time")
plt.ylabel("Solar Generation (MW)")
save_current_figure()

# Seasonal patterns

# Add seasonal time variables
solar["hour"] = solar["ds"].dt.hour
solar["month"] = solar["ds"].dt.month

# Typical 24-hour profile
hourly_profile = solar.groupby("hour")["y"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_profile.index, hourly_profile.values, marker="o")
plt.title("Average Solar Generation by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Average Solar Generation (MW)")
plt.xticks(range(24))
save_current_figure()

# Average generation by month
monthly_profile = solar.groupby("month")["y"].mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly_profile.index, monthly_profile.values, marker="o")
plt.title("Average Solar Generation by Month")
plt.xlabel("Month")
plt.ylabel("Average Solar Generation (MW)")
plt.xticks(range(1, 13))
save_current_figure()

# Compare the typical daily curve across representative seasons
month_names = {1: "January", 4: "April", 7: "July", 10: "October"}

plt.figure(figsize=(12, 6))

for month, name in month_names.items():
    profile = solar[solar["month"] == month].groupby("hour")["y"].mean()
    plt.plot(profile.index, profile.values, marker="o", label=name)

plt.title("Average Daily Solar Generation Across Seasons")
plt.xlabel("Hour of Day")
plt.ylabel("Average Solar Generation (MW)")
plt.xticks(range(24))
plt.legend()
save_current_figure()

# Autocorrelation

from statsmodels.graphics.tsaplots import plot_acf

# One month is enough to clearly show daily autocorrelation
acf_data = solar[(solar["ds"] >= "2019-06-01") & (solar["ds"] < "2019-07-01")]["y"].dropna()

fig, ax = plt.subplots(figsize=(14, 5))
plot_acf(acf_data, lags=168, ax=ax)

for lag in [24, 48, 72, 96, 120, 144, 168]:
    ax.axvline(lag, linestyle="--", alpha=0.4)

plt.title("ACF of Hourly Solar Generation")
plt.xlabel("Lag (Hours)")
plt.ylabel("Autocorrelation")
save_current_figure()

# Seasonal naive baseline

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Keep the original hourly rows so shift(24) really means 24 hours earlier
baseline = solar[["ds", "y"]].copy()
baseline["naive_forecast"] = baseline["y"].shift(24)

# Evaluate on the complete year 2019
baseline_test = baseline[
    (baseline["ds"] >= "2019-01-01") &
    (baseline["ds"] < "2020-01-01")
].dropna(subset=["y", "naive_forecast"]).copy()

mae_naive = mean_absolute_error(baseline_test["y"], baseline_test["naive_forecast"])
rmse_naive = np.sqrt(mean_squared_error(baseline_test["y"], baseline_test["naive_forecast"]))

print("Seasonal-Naive Baseline")
print(f"MAE  : {mae_naive:.2f} MW")
print(f"RMSE : {rmse_naive:.2f} MW")

# Visual check of the baseline for one week
baseline_week = baseline_test[
    (baseline_test["ds"] >= "2019-06-03") &
    (baseline_test["ds"] < "2019-06-10")
]

plt.figure(figsize=(16, 6))
plt.plot(baseline_week["ds"], baseline_week["y"], label="Actual")
plt.plot(baseline_week["ds"], baseline_week["naive_forecast"], label="Seasonal Naive", linestyle="--")
plt.title("Actual vs Seasonal-Naive Forecast")
plt.xlabel("Time")
plt.ylabel("Solar Generation (MW)")
plt.legend()
save_current_figure()

# Prophet models

from prophet import Prophet

# Prophet input
prophet_data = solar[["ds", "y"]].dropna().copy()
prophet_data["ds"] = prophet_data["ds"].dt.tz_convert(None)

# Train on the past, test on the full year 2019
train = prophet_data[prophet_data["ds"] < "2019-01-01"].copy()
test = prophet_data[(prophet_data["ds"] >= "2019-01-01") & (prophet_data["ds"] < "2020-01-01")].copy()

print("Training:", train["ds"].min(), "to", train["ds"].max())
print("Testing :", test["ds"].min(), "to", test["ds"].max())

# MODEL 1 — Trend only
model_1 = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
model_1.fit(train)
forecast_1 = model_1.predict(test[["ds"]])

mae_model_1 = mean_absolute_error(test["y"], forecast_1["yhat"])
rmse_model_1 = np.sqrt(mean_squared_error(test["y"], forecast_1["yhat"]))

print("Trend Only")
print(f"MAE  : {mae_model_1:.2f} MW")
print(f"RMSE : {rmse_model_1:.2f} MW")

# MODEL 2 — Trend + daily seasonality
model_2 = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
model_2.fit(train)
forecast_2 = model_2.predict(test[["ds"]])

mae_model_2 = mean_absolute_error(test["y"], forecast_2["yhat"])
rmse_model_2 = np.sqrt(mean_squared_error(test["y"], forecast_2["yhat"]))

print("Daily Seasonality")
print(f"MAE  : {mae_model_2:.2f} MW")
print(f"RMSE : {rmse_model_2:.2f} MW")

# MODEL 3 — Trend + daily + yearly seasonality
model_3 = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=True)
model_3.fit(train)
forecast_3 = model_3.predict(test[["ds"]])

results_3 = test[["ds", "y"]].copy()
results_3["yhat"] = forecast_3["yhat"].values

mae_model_3 = mean_absolute_error(results_3["y"], results_3["yhat"])
rmse_model_3 = np.sqrt(mean_squared_error(results_3["y"], results_3["yhat"]))

print("Daily + Yearly Seasonality")
print(f"MAE  : {mae_model_3:.2f} MW")
print(f"RMSE : {rmse_model_3:.2f} MW")

# Final Prophet forecast for one representative week
results_week = results_3[(results_3["ds"] >= "2019-06-03") & (results_3["ds"] < "2019-06-10")]

plt.figure(figsize=(16, 6))
plt.plot(results_week["ds"], results_week["y"], label="Actual")
plt.plot(results_week["ds"], results_week["yhat"], label="Prophet Daily + Yearly", linestyle="--")
plt.title("Actual vs Multi-Seasonal Prophet Forecast")
plt.xlabel("Time")
plt.ylabel("Solar Generation (MW)")
plt.legend()
save_current_figure()

# Show the trend, daily and yearly components learned by Prophet
model_3.plot_components(forecast_3)
save_current_figure()

# Model comparison

final_comparison = pd.DataFrame({
    "Model": [
        "Seasonal Naive",
        "Prophet Trend Only",
        "Prophet Daily",
        "Prophet Daily + Yearly"
    ],
    "MAE (MW)": [mae_naive, mae_model_1, mae_model_2, mae_model_3],
    "RMSE (MW)": [rmse_naive, rmse_model_1, rmse_model_2, rmse_model_3]
})

print(final_comparison)

best_model = final_comparison.loc[final_comparison["RMSE (MW)"].idxmin()]
print("\nLowest-RMSE model:", best_model["Model"])
print("RMSE:", round(best_model["RMSE (MW)"], 2), "MW")

plt.figure(figsize=(10, 5))
plt.bar(final_comparison["Model"], final_comparison["RMSE (MW)"])
plt.title("Forecasting Model Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE (MW)")
plt.xticks(rotation=20)
save_current_figure()

# Residual checks

# Residual = Actual - Predicted
results_3["residual"] = results_3["y"] - results_3["yhat"]

residual_sample = results_3[
    (results_3["ds"] >= "2019-06-01") &
    (results_3["ds"] < "2019-07-01")
]["residual"].dropna()

fig, ax = plt.subplots(figsize=(14, 5))
plot_acf(residual_sample, lags=168, ax=ax)

for lag in [24, 48, 72, 96, 120, 144, 168]:
    ax.axvline(lag, linestyle="--", alpha=0.4)

plt.title("ACF of Prophet Residuals")
plt.xlabel("Lag (Hours)")
plt.ylabel("Autocorrelation")
save_current_figure()

# Find when during the day Prophet makes the largest errors
results_3["hour"] = results_3["ds"].dt.hour
results_3["absolute_error"] = abs(results_3["y"] - results_3["yhat"])

error_by_hour = results_3.groupby("hour")["absolute_error"].mean()

plt.figure(figsize=(10, 5))
plt.plot(error_by_hour.index, error_by_hour.values, marker="o")
plt.title("Average Prophet Forecast Error by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Mean Absolute Error (MW)")
plt.xticks(range(24))
save_current_figure()

final_comparison.to_csv("model_comparison.csv", index=False)
results_3.to_csv("prophet_predictions_2019.csv", index=False)
print("Analysis complete. Graphs saved in figures/.")

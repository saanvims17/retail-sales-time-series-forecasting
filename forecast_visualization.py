import pandas as pd
import matplotlib.pyplot as plt

HISTORY_PATH = "data/retail_sales_prepared.csv"
FORECAST_PATH = "data/all_store_forecasts.csv"

STORE = 1
DEPT = 1

history = pd.read_csv(HISTORY_PATH)
forecast = pd.read_csv(FORECAST_PATH)

history["Date"] = pd.to_datetime(history["Date"])
forecast["Date"] = pd.to_datetime(forecast["Date"])

history = history[
    (history["Store"] == STORE) &
    (history["Dept"] == DEPT)
].sort_values("Date")

forecast = forecast[
    (forecast["Store"] == STORE) &
    (forecast["Dept"] == DEPT)
].sort_values("Date")

plt.figure(figsize=(12, 6))

plt.plot(
    history["Date"].tail(52),
    history["Weekly_Sales"].tail(52),
    label="Historical Sales"
)

plt.plot(
    forecast["Date"],
    forecast["Forecast"],
    marker="o",
    label="Forecast"
)

plt.axvline(
    history["Date"].max(),
    linestyle="--",
    label="Forecast Start"
)

plt.title(
    f"Retail Sales Forecast — Store {STORE}, Department {DEPT}"
)

plt.xlabel("Date")
plt.ylabel("Weekly Sales")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "data/forecast_visualization.png",
    dpi=150
)

plt.show()
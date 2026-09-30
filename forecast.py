import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "xgboost_retail_model.pkl"
DATA_PATH = "data/retail_sales_features.csv"

FORECAST_WEEKS = 8

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Store", "Dept", "Date"]
).reset_index(drop=True)

forecasts = []

groups = df.groupby(["Store", "Dept"])

print(f"Total Store-Department series: {len(groups)}")
print("Generating forecasts...\n")

for (store, dept), series in groups:

    series = series.sort_values("Date").copy()

    if len(series) < 52:
        continue

    history = series[
        ["Date", "Weekly_Sales"]
    ].copy()

    last_row = series.iloc[-1]

    future_dates = pd.date_range(
        start=history["Date"].max() + pd.Timedelta(weeks=1),
        periods=FORECAST_WEEKS,
        freq="7D"
    )

    for future_date in future_dates:

        sales = history["Weekly_Sales"]

        features = {
            "Store": store,
            "Dept": dept,
            "IsHoliday": 0,
            "Temperature": last_row["Temperature"],
            "Fuel_Price": last_row["Fuel_Price"],
            "MarkDown1": 0,
            "MarkDown2": 0,
            "MarkDown3": 0,
            "MarkDown4": 0,
            "MarkDown5": 0,
            "CPI": last_row["CPI"],
            "Unemployment": last_row["Unemployment"],
            "Size": last_row["Size"],
            "Year": future_date.year,
            "Month": future_date.month,
            "Week": future_date.isocalendar().week,
            "Quarter": future_date.quarter,
            "DayOfYear": future_date.dayofyear,
            "Lag_1": sales.iloc[-1],
            "Lag_4": sales.iloc[-4],
            "Lag_12": sales.iloc[-12],
            "Lag_52": sales.iloc[-52],
            "Rolling_Mean_4": sales.iloc[-4:].mean(),
            "Rolling_Mean_12": sales.iloc[-12:].mean(),
            "Rolling_Std_4": sales.iloc[-4:].std(),
            "Type_A": int(last_row["Type"] == "A"),
            "Type_B": int(last_row["Type"] == "B"),
            "Type_C": int(last_row["Type"] == "C")
        }

        X_future = pd.DataFrame([features])
        X_future = X_future[model.feature_names_in_]

        prediction = model.predict(X_future)[0]

        forecasts.append({
            "Store": store,
            "Dept": dept,
            "Date": future_date,
            "Forecast": max(0, prediction)
        })

        history = pd.concat(
            [
                history,
                pd.DataFrame({
                    "Date": [future_date],
                    "Weekly_Sales": [prediction]
                })
            ],
            ignore_index=True
        )

forecast_df = pd.DataFrame(forecasts)

forecast_df = forecast_df.sort_values(
    ["Store", "Dept", "Date"]
)

forecast_df.to_csv(
    "data/all_store_forecasts.csv",
    index=False
)

print("Forecast completed!")
print(f"Forecast rows: {len(forecast_df):,}")
print(f"Store-Department series: {forecast_df.groupby(['Store', 'Dept']).ngroups}")

print("\nSample forecasts:")
print(
    forecast_df.head(20).to_string(index=False)
)

print("\nForecast saved to:")
print("data/all_store_forecasts.csv")
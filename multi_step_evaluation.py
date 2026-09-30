import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_PATH = "xgboost_retail_model.pkl"
DATA_PATH = "data/retail_sales_features.csv"

FORECAST_WEEKS = 8

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Store", "Dept", "Date"]
).reset_index(drop=True)

predictions = []

groups = df.groupby(["Store", "Dept"])

print(f"Total series: {len(groups)}")
print("Running 8-week recursive evaluation...\n")

for (store, dept), series in groups:

    series = series.sort_values("Date").reset_index(drop=True)

    if len(series) < 60:
        continue

    evaluation = series.tail(FORECAST_WEEKS).copy()
    history = series.iloc[:-FORECAST_WEEKS].copy()

    if len(history) < 52:
        continue

    sales_history = history["Weekly_Sales"].tolist()

    for _, future_row in evaluation.iterrows():

        sales = pd.Series(sales_history)

        features = {
            "Store": store,
            "Dept": dept,
            "IsHoliday": future_row["IsHoliday"],
            "Temperature": future_row["Temperature"],
            "Fuel_Price": future_row["Fuel_Price"],
            "MarkDown1": future_row["MarkDown1"],
            "MarkDown2": future_row["MarkDown2"],
            "MarkDown3": future_row["MarkDown3"],
            "MarkDown4": future_row["MarkDown4"],
            "MarkDown5": future_row["MarkDown5"],
            "CPI": future_row["CPI"],
            "Unemployment": future_row["Unemployment"],
            "Size": future_row["Size"],
            "Year": future_row["Date"].year,
            "Month": future_row["Date"].month,
            "Week": future_row["Date"].isocalendar().week,
            "Quarter": future_row["Date"].quarter,
            "DayOfYear": future_row["Date"].dayofyear,
            "Lag_1": sales.iloc[-1],
            "Lag_4": sales.iloc[-4],
            "Lag_12": sales.iloc[-12],
            "Lag_52": sales.iloc[-52],
            "Rolling_Mean_4": sales.iloc[-4:].mean(),
            "Rolling_Mean_12": sales.iloc[-12:].mean(),
            "Rolling_Std_4": sales.iloc[-4:].std(),
            "Type_A": int(future_row["Type"] == "A"),
            "Type_B": int(future_row["Type"] == "B"),
            "Type_C": int(future_row["Type"] == "C")
        }

        X_future = pd.DataFrame([features])
        X_future = X_future[model.feature_names_in_]

        prediction = model.predict(X_future)[0]
        prediction = max(0, prediction)

        predictions.append({
            "Store": store,
            "Dept": dept,
            "Date": future_row["Date"],
            "Actual": future_row["Weekly_Sales"],
            "Predicted": prediction
        })

        sales_history.append(prediction)

results = pd.DataFrame(predictions)

mae = mean_absolute_error(
    results["Actual"],
    results["Predicted"]
)

rmse = np.sqrt(
    mean_squared_error(
        results["Actual"],
        results["Predicted"]
    )
)

wape = (
    np.sum(
        np.abs(
            results["Actual"] - results["Predicted"]
        )
    )
    / np.sum(np.abs(results["Actual"]))
) * 100

smape = (
    np.mean(
        2 * np.abs(
            results["Actual"] - results["Predicted"]
        )
        /
        (
            np.abs(results["Actual"])
            + np.abs(results["Predicted"])
            + 1e-8
        )
    )
) * 100

print("\nMulti-Step Forecast Evaluation")
print("=" * 40)

print(f"Forecast horizon : {FORECAST_WEEKS} weeks")
print(f"Series evaluated  : {results.groupby(['Store', 'Dept']).ngroups}")
print(f"Predictions       : {len(results):,}")

print("\nMetrics")
print("-" * 40)
print(f"MAE   : {mae:,.2f}")
print(f"RMSE  : {rmse:,.2f}")
print(f"WAPE  : {wape:.2f}%")
print(f"sMAPE : {smape:.2f}%")

results.to_csv(
    "data/multi_step_evaluation.csv",
    index=False
)

print("\nResults saved to:")
print("data/multi_step_evaluation.csv")
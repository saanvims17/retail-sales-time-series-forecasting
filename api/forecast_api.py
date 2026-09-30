import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI(title="Retail Demand Forecasting API")

model = joblib.load("vertex_model/model.pkl")

DATA_PATH = "data/retail_sales_prepared.csv"

FEATURES = [
    "Store", "Dept", "IsHoliday", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "Size", "Year", "Month", "Week",
    "Quarter", "DayOfYear", "Lag_1", "Lag_4", "Lag_12", "Lag_52",
    "Rolling_Mean_4", "Rolling_Mean_12", "Rolling_Std_4",
    "Type_A", "Type_B", "Type_C"
]

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values(["Store", "Dept", "Date"])


class ForecastRequest(BaseModel):
    store: int
    department: int
    weeks: int = 8


def is_holiday(date):
    holidays = [
        "2012-11-23",
        "2012-12-25",
        "2013-01-01"
    ]
    return int(date.strftime("%Y-%m-%d") in holidays)


def create_features(history, date):
    latest = history.iloc[-1]

    values = {
        "Store": int(latest["Store"]),
        "Dept": int(latest["Dept"]),
        "IsHoliday": is_holiday(date),
        "Temperature": float(latest["Temperature"]),
        "Fuel_Price": float(latest["Fuel_Price"]),
        "MarkDown1": float(latest["MarkDown1"]),
        "MarkDown2": float(latest["MarkDown2"]),
        "MarkDown3": float(latest["MarkDown3"]),
        "MarkDown4": float(latest["MarkDown4"]),
        "MarkDown5": float(latest["MarkDown5"]),
        "CPI": float(latest["CPI"]),
        "Unemployment": float(latest["Unemployment"]),
        "Size": float(latest["Size"]),
        "Year": date.year,
        "Month": date.month,
        "Week": int(date.isocalendar().week),
        "Quarter": date.quarter,
        "DayOfYear": date.dayofyear,
        "Lag_1": float(history.iloc[-1]["Weekly_Sales"]),
        "Lag_4": float(history.iloc[-4]["Weekly_Sales"]),
        "Lag_12": float(history.iloc[-12]["Weekly_Sales"]),
        "Lag_52": float(history.iloc[-52]["Weekly_Sales"]),
        "Rolling_Mean_4": float(history["Weekly_Sales"].tail(4).mean()),
        "Rolling_Mean_12": float(history["Weekly_Sales"].tail(12).mean()),
        "Rolling_Std_4": float(history["Weekly_Sales"].tail(4).std()),
        "Type_A": int(latest["Type"] == "A"),
        "Type_B": int(latest["Type"] == "B"),
        "Type_C": int(latest["Type"] == "C")
    }

    return pd.DataFrame([values])[FEATURES]


@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecasting API",
        "status": "running"
    }


@app.get("/app")
def frontend():
    return FileResponse("frontend/index.html")

@app.post("/forecast")
def forecast(request: ForecastRequest):

    history = df[
        (df["Store"] == request.store) &
        (df["Dept"] == request.department)
    ].copy()

    if history.empty:
        return {"error": "Store and department combination not found"}

    predictions = []

    last_date = history["Date"].max()

    for i in range(1, request.weeks + 1):

        future_date = last_date + pd.Timedelta(weeks=i)

        features = create_features(history, future_date)

        prediction = float(model.predict(features)[0])

        predictions.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "predicted_weekly_sales": round(prediction, 2)
        })

        new_row = history.iloc[-1].copy()
        new_row["Date"] = future_date
        new_row["Weekly_Sales"] = prediction

        history = pd.concat(
            [history, pd.DataFrame([new_row])],
            ignore_index=True
        )

    return {
        "store": request.store,
        "department": request.department,
        "forecast_weeks": request.weeks,
        "predictions": predictions
    }
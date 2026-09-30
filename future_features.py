import pandas as pd
import numpy as np


HOLIDAY_DATES = {
    "Super_Bowl": [
        "2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"
    ],
    "Labor_Day": [
        "2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"
    ],
    "Thanksgiving": [
        "2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"
    ],
    "Christmas": [
        "2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"
    ]
}


def get_holiday(date):
    date = pd.Timestamp(date).strftime("%Y-%m-%d")

    for holiday, dates in HOLIDAY_DATES.items():
        if date in dates:
            return 1

    return 0


def get_markdown_estimates(df, store, month):
    store_data = df[
        (df["Store"] == store) &
        (df["Month"] == month)
    ]

    markdown_columns = [
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5"
    ]

    estimates = {}

    for column in markdown_columns:
        value = store_data[column].median()

        if pd.isna(value):
            value = 0

        estimates[column] = value

    return estimates


def build_future_features(df, store, dept, future_date):

    series = df[
        (df["Store"] == store) &
        (df["Dept"] == dept)
    ].sort_values("Date")

    last_row = series.iloc[-1]

    month = future_date.month

    markdowns = get_markdown_estimates(
        df,
        store,
        month
    )

    features = {
        "Store": store,
        "Dept": dept,

        "IsHoliday": get_holiday(future_date),

        "Temperature": last_row["Temperature"],
        "Fuel_Price": last_row["Fuel_Price"],

        "MarkDown1": markdowns["MarkDown1"],
        "MarkDown2": markdowns["MarkDown2"],
        "MarkDown3": markdowns["MarkDown3"],
        "MarkDown4": markdowns["MarkDown4"],
        "MarkDown5": markdowns["MarkDown5"],

        "CPI": last_row["CPI"],
        "Unemployment": last_row["Unemployment"],

        "Size": last_row["Size"],

        "Year": future_date.year,
        "Month": future_date.month,
        "Week": future_date.isocalendar().week,
        "Quarter": future_date.quarter,
        "DayOfYear": future_date.dayofyear,

        "Type_A": int(last_row["Type"] == "A"),
        "Type_B": int(last_row["Type"] == "B"),
        "Type_C": int(last_row["Type"] == "C")
    }

    return features
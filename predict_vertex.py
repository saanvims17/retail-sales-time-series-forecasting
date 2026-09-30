import pandas as pd
from google.cloud import aiplatform

PROJECT_ID = "linen-compiler-510211-c2"
REGION = "us-central1"
ENDPOINT_ID = "5698588996611866624"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION
)

endpoint = aiplatform.Endpoint(
    endpoint_name=ENDPOINT_ID
)

df = pd.read_csv("data/test_features.csv")

row = df.iloc[0].copy()

input_df = df.drop(columns=["Weekly_Sales", "Date"]).iloc[[0]].copy()

input_df = pd.get_dummies(
    input_df,
    columns=["Type"]
)

expected_columns = [
    "Store",
    "Dept",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
    "Size",
    "Year",
    "Month",
    "Week",
    "Quarter",
    "DayOfYear",
    "Lag_1",
    "Lag_4",
    "Lag_12",
    "Lag_52",
    "Rolling_Mean_4",
    "Rolling_Mean_12",
    "Rolling_Std_4",
    "Type_A",
    "Type_B",
    "Type_C"
]

input_df = input_df.reindex(
    columns=expected_columns,
    fill_value=0
)

features = input_df.iloc[0].astype(float).tolist()

response = endpoint.predict(
    instances=[features]
)

actual = float(row["Weekly_Sales"])
prediction = float(response.predictions[0])

print("\nVertex AI Prediction")
print("--------------------")
print(f"Actual Weekly Sales:    {actual:,.2f}")
print(f"Predicted Weekly Sales: {prediction:,.2f}")
print(f"Difference:             {abs(actual - prediction):,.2f}")
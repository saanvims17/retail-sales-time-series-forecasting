import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load datasets
train = pd.read_csv("data/train_features.csv")
validation = pd.read_csv("data/validation_features.csv")
test = pd.read_csv("data/test_features.csv")

train["Date"] = pd.to_datetime(train["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])
test["Date"] = pd.to_datetime(test["Date"])


# Naive Forecast
# Predict the current week's sales using the previous week's sales.

validation_predictions = validation["Lag_1"]

test_predictions = test["Lag_1"]


# Actual values
validation_actual = validation["Weekly_Sales"]
test_actual = test["Weekly_Sales"]


# Evaluation function

def evaluate_model(actual, predicted, name):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # Avoid division by zero
    non_zero = actual != 0

    mape = np.mean(
        np.abs(
            (actual[non_zero] - predicted[non_zero])
            / actual[non_zero]
        )
    ) * 100

    print(f"\n{name}")
    print("-" * 30)
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"MAPE : {mape:.2f}%")


# Evaluate validation set
evaluate_model(
    validation_actual,
    validation_predictions,
    "Naive Validation"
)


# Evaluate test set
evaluate_model(
    test_actual,
    test_predictions,
    "Naive Test"
)
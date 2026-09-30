import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load test data
test = pd.read_csv("data/test_features.csv")

test["Date"] = pd.to_datetime(test["Date"])


# Load trained model
model = joblib.load("xgboost_retail_model.pkl")


# Prepare features
test_model = pd.get_dummies(
    test,
    columns=["Type"]
)

X_test = test_model.drop(
    columns=["Weekly_Sales", "Date"]
)

y_test = test["Weekly_Sales"]


# Make predictions
predictions = model.predict(X_test)


# ------------------------------------------------------------
# Additional evaluation metrics
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# WAPE
wape = (
    np.sum(np.abs(y_test - predictions))
    / np.sum(np.abs(y_test))
) * 100


# sMAPE
smape = (
    np.mean(
        2 * np.abs(y_test - predictions)
        /
        (
            np.abs(y_test)
            + np.abs(predictions)
            + 1e-8
        )
    )
) * 100


print("\nModel Evaluation")
print("------------------------------")
print(f"MAE   : {mae:,.2f}")
print(f"RMSE  : {rmse:,.2f}")
print(f"WAPE  : {wape:.2f}%")
print(f"sMAPE : {smape:.2f}%")


# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X_test.columns
)

importance = importance.sort_values(
    ascending=False
)


print("\nTop 15 Features")
print("------------------------------")
print(importance.head(15))


# Plot top 15 features
plt.figure(figsize=(10, 6))

importance.head(15).sort_values().plot(
    kind="barh"
)

plt.title("Top 15 XGBoost Features")
plt.xlabel("Importance")

plt.tight_layout()
plt.show()
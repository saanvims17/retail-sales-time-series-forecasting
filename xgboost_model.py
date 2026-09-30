import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load datasets
train = pd.read_csv("data/train_features.csv")
validation = pd.read_csv("data/validation_features.csv")
test = pd.read_csv("data/test_features.csv")

# Convert dates
for df in [train, validation, test]:
    df["Date"] = pd.to_datetime(df["Date"])

# Columns that should not be used directly as model features
target = "Weekly_Sales"
drop_columns = ["Weekly_Sales", "Date"]

# Convert Store Type into numerical columns
train = pd.get_dummies(train, columns=["Type"])
validation = pd.get_dummies(validation, columns=["Type"])
test = pd.get_dummies(test, columns=["Type"])

# Make sure all datasets have the same columns
validation = validation.reindex(
    columns=train.columns,
    fill_value=0
)

test = test.reindex(
    columns=train.columns,
    fill_value=0
)

# Separate features and target
X_train = train.drop(columns=drop_columns)
y_train = train[target]

X_validation = validation.drop(columns=drop_columns)
y_validation = validation[target]

X_test = test.drop(columns=drop_columns)
y_test = test[target]

print("Training shape:", X_train.shape)
print("Validation shape:", X_validation.shape)
print("Test shape:", X_test.shape)

# Create XGBoost model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    n_jobs=-1,
    random_state=42
)

# Train
print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=False
)

print("Training completed!")

# Predictions
validation_predictions = model.predict(X_validation)
test_predictions = model.predict(X_test)

# Evaluation function
def evaluate_model(actual, predicted, name):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # MAPE can become unstable when actual sales
    # are very small.
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


# Evaluate validation
evaluate_model(
    y_validation,
    validation_predictions,
    "XGBoost Validation"
)


# Evaluate test
evaluate_model(
    y_test,
    test_predictions,
    "XGBoost Test"
)


# Save model
joblib.dump(
    model,
    "xgboost_retail_model.pkl"
)

print("\nModel saved as:")
print("xgboost_retail_model.pkl")
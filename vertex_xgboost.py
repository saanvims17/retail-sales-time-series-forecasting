import argparse
import os

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--model-dir", required=True)

    args = parser.parse_args()

    train = pd.read_csv(args.train_data)
    validation = pd.read_csv(args.validation_data)

    train["Date"] = pd.to_datetime(train["Date"])
    validation["Date"] = pd.to_datetime(validation["Date"])

    train = pd.get_dummies(train, columns=["Type"])
    validation = pd.get_dummies(validation, columns=["Type"])

    validation = validation.reindex(
        columns=train.columns,
        fill_value=0
    )

    target = "Weekly_Sales"
    drop_columns = ["Weekly_Sales", "Date"]

    X_train = train.drop(columns=drop_columns)
    y_train = train[target]

    X_validation = validation.drop(columns=drop_columns)
    y_validation = validation[target]

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_validation.shape)

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

    print("Training XGBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False
    )

    predictions = model.predict(X_validation)

    mae = mean_absolute_error(
        y_validation,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions
        )
    )

    print("Training completed!")
    print(f"Validation MAE: {mae:,.2f}")
    print(f"Validation RMSE: {rmse:,.2f}")

    os.makedirs(args.model_dir, exist_ok=True)

    model_path = os.path.join(
        args.model_dir,
        "xgboost_retail_model.pkl"
    )

    joblib.dump(model, model_path)

    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()
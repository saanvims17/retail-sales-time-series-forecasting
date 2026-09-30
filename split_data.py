import pandas as pd


# Load feature-engineered data
df = pd.read_csv("data/retail_sales_features.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Make sure data is chronological
df = df.sort_values("Date").reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Start date:", df["Date"].min())
print("End date:", df["Date"].max())


# Time-based split
train = df[df["Date"] < "2012-01-01"]

validation = df[
    (df["Date"] >= "2012-01-01") &
    (df["Date"] < "2012-07-01")
]

test = df[df["Date"] >= "2012-07-01"]


# Display sizes
print("\nTrain:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)


# Display date ranges
print("\nTrain period:")
print(train["Date"].min(), "→", train["Date"].max())

print("\nValidation period:")
print(validation["Date"].min(), "→", validation["Date"].max())

print("\nTest period:")
print(test["Date"].min(), "→", test["Date"].max())


# Save datasets
train.to_csv(
    "data/train_features.csv",
    index=False
)

validation.to_csv(
    "data/validation_features.csv",
    index=False
)

test.to_csv(
    "data/test_features.csv",
    index=False
)

print("\nDatasets saved successfully!")
import pandas as pd

# Load datasets
train = pd.read_csv("data/train.csv")
features = pd.read_csv("data/features.csv")
stores = pd.read_csv("data/stores.csv")

print("Train:", train.shape)
print("Features:", features.shape)
print("Stores:", stores.shape)

# Convert Date
train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])

# Merge datasets
df = train.merge(
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)

df = df.merge(
    stores,
    on="Store",
    how="left"
)

# Fill markdown missing values
markdown_columns = [
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5"
]

df[markdown_columns] = df[markdown_columns].fillna(0)

# Sort chronologically
df = df.sort_values(
    ["Store", "Dept", "Date"]
).reset_index(drop=True)

# Save
df.to_csv(
    "data/retail_sales_prepared.csv",
    index=False
)

print("Prepared dataset saved!")
print("Final shape:", df.shape)
print(df.head())
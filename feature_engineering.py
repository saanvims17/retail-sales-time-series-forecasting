import pandas as pd


# Load data
df = pd.read_csv("data/retail_sales_prepared.csv")
df["Date"] = pd.to_datetime(df["Date"])


# Sort chronologically
df = df.sort_values(
    ["Store", "Dept", "Date"]
).reset_index(drop=True)

print("Initial shape:", df.shape)


# Calendar features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["Quarter"] = df["Date"].dt.quarter
df["DayOfYear"] = df["Date"].dt.dayofyear


# Lag features

# Lag_1: Sales from the previous week for the same Store and Department.
df["Lag_1"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .shift(1)
)

# Lag_4: Sales approximately 4 weeks ago.
df["Lag_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .shift(4)
)

# Lag_12: Sales approximately 12 weeks ago.
df["Lag_12"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .shift(12)
)

# Lag_52: Sales approximately one year ago.
df["Lag_52"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .shift(52)
)


# Rolling features

# Rolling_Mean_4: Average sales over the previous 4 weeks.
df["Rolling_Mean_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(
        lambda x: x.shift(1).rolling(4).mean()
    )
)

# Rolling_Mean_12: Average sales over the previous 12 weeks.
df["Rolling_Mean_12"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(
        lambda x: x.shift(1).rolling(12).mean()
    )
)

# Rolling_Std_4 measures sales volatility, over the previous 4 weeks.
df["Rolling_Std_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(
        lambda x: x.shift(1).rolling(4).std()
    )
)


# Remove rows without enough historical data
df = df.dropna(
    subset=[
        "Lag_1",
        "Lag_4",
        "Lag_12",
        "Lag_52",
        "Rolling_Mean_4",
        "Rolling_Mean_12",
        "Rolling_Std_4"
    ]
).reset_index(drop=True)


# Check final dataset
print("Final shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())


# Save
df.to_csv(
    "data/retail_sales_features.csv",
    index=False
)

print("\nFeature-engineered dataset saved!")

# DISPLAY 

print("\nFeature preview:")

print(
    df[
        [
            "Store",
            "Dept",
            "Date",
            "Weekly_Sales",
            "Lag_1",
            "Lag_4",
            "Lag_12",
            "Lag_52",
            "Rolling_Mean_4",
            "Rolling_Mean_12",
            "Rolling_Std_4"
        ]
    ].head(60)
)

 # CHECK MISSING VALUES
 
print("\nMissing values before removing insufficient history:")

print(
    df[
        [
            "Lag_1",
            "Lag_4",
            "Lag_12",
            "Lag_52",
            "Rolling_Mean_4",
            "Rolling_Mean_12",
            "Rolling_Std_4"
        ]
    ].isnull().sum()
)

# REMOVE ROWS WITHOUT ENOUGH HISTORY

df = df.dropna(
    subset=[
        "Lag_1",
        "Lag_4",
        "Lag_12",
        "Lag_52",
        "Rolling_Mean_4",
        "Rolling_Mean_12",
        "Rolling_Std_4"
    ]
).reset_index(drop=True)

# CHECK FINAL DATASET

print("\nFinal dataset shape:", df.shape)

print("\nRemaining missing values:")

print(df.isnull().sum())

# SAVE FEATURE-ENGINEERED DATASET

df.to_csv(
    "data/retail_sales_features.csv",
    index=False
)

print(
    "\nFeature-engineered dataset saved successfully!"
)

print(
    "File: data/retail_sales_features.csv"
)
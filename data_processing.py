import pandas as pd

# Load data and turn it into one single dataframe
df_1 = pd.read_csv("data/transactions_1.csv", index_col=0)
df_2 = pd.read_csv("data/transactions_2.csv", index_col=0)
df = pd.concat([df_1, df_2])

# Standardize product names
df["product_id"] = df["product_id"].replace("├ÅTS", "ATS").replace("MCC/Smart", "Smart")
df = df[~df['product_id'].isin(["Not a make", "Undefined"])]

# Convert 'date' column to datetime
df["date"] = pd.to_datetime(df["date"])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df["year_month"] = df["date"].dt.to_period("M")

df.to_csv("data/all_transactions.csv", index=False)


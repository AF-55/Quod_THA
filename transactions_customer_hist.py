import pandas as pd
import matplotlib.pyplot as plt

# Loading transactions dataset
df = pd.read_csv("data/all_transactions.csv", index_col=0)

# Sorting total number of transactions per customer over the entier period
customer_monthly = df.groupby(['customer_id', 'year_month']).size().reset_index(name='transactions')
customer_counts = customer_monthly.groupby('customer_id')['transactions'].sum().sort_values(ascending=False)

# Saving plots as a CSV file
customer_counts.to_csv("csv_results/transactions_customer_hist")

# Plotting and saving plots
plt.figure(figsize=(12, 8))
plt.bar(customer_counts.index.astype(str), customer_counts.values, width=1, color='orange')
plt.xlabel("Customers from best to worst buyer")
plt.ylabel("Number of Transactions")
plt.title("Transactions per Customer (Descending Order)")
plt.xticks([])
plt.yscale('log')
plt.grid(linestyle='--', color='black')
plt.savefig("plots/transactions_customer_hist.png")
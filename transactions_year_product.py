import pandas as pd
import matplotlib.pyplot as plt

year = 2018
product = "Volvo"

df = pd.read_csv("data/all_transactions.csv", index_col=0)

product_monthly = df.groupby(['product_id', 'year_month']).size().reset_index(name='transactions')
product_monthly['year_month'] = pd.to_datetime(product_monthly['year_month'], format='%Y-%m')
tr = product_monthly[(product_monthly['year_month'].dt.year == year) & (product_monthly['product_id'] == product)]

plt.figure(figsize=(12,8))
plt.plot(range(len(tr)), tr['transactions'].values, marker='o', color='royalblue', linestyle='dashed')
plt.grid('--')
ticks = tr.year_month.dt.to_period('M')
plt.xticks(ticks=range(len(tr)), labels=ticks)
plt.xlabel('Month')
plt.ylabel('Number of transactions')
plt.title(f'Transaction frequency per month for {product} between {ticks.values[0]} and {ticks.values[-1]}')
plt.savefig("results/transactions_year_product.png")


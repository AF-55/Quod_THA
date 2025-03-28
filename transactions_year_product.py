import pandas as pd
import matplotlib.pyplot as plt

# Loading transactions dataset
df = pd.read_csv("data/all_transactions.csv", index_col=0)

# We will create a plot showing the transaction frequency per month for the year 2018 of a certain product
year = 2018
product = "Volvo"

# Monthly transactions for the specific product over the year
product_monthly = df.groupby(['product_id', 'year_month']).size().reset_index(name='transactions')
product_monthly['year_month'] = pd.to_datetime(product_monthly['year_month'], format='%Y-%m')
this_product_monthly = product_monthly[(product_monthly['year_month'].dt.year == year) & (product_monthly['product_id'] == product)]

# Saving the plots as a CSV file
this_product_monthly.reset_index(drop=True).to_csv("csv_results/transactions_year_product.csv")

# Plotting and saving the plots
plt.figure(figsize=(12,8))
plt.plot(range(len(this_product_monthly)), this_product_monthly['transactions'].values, marker='o', color='royalblue', linestyle='dashed')
plt.grid(linestyle='--')
ticks = this_product_monthly.year_month.dt.to_period('M')
plt.xticks(ticks=range(len(this_product_monthly)), labels=ticks)
plt.xlabel('Month')
plt.ylabel('Number of transactions')
plt.title(f'Transaction frequency per month for {product} between {ticks.values[0]} and {ticks.values[-1]}')
plt.savefig("plots/transactions_year_product.png")


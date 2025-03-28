import pandas as pd

df = pd.read_csv("data/all_transactions.csv", index_col=0)

reference_date = '2018-01-01'
nb_months = 6
nb_top = 5

reference_date = pd.to_datetime(reference_date).tz_localize('UTC')
six_months_ago = reference_date - pd.DateOffset(months=nb_months)
reference_date = reference_date.strftime('%Y-%m-%d')
six_months_ago = six_months_ago.strftime('%Y-%m-%d')

formatted_dates = df['date'].apply(lambda x:x[:10])
df_recent = df[(formatted_dates >= six_months_ago) & (formatted_dates <= reference_date)]
top_products = df_recent['product_id'].value_counts().head(nb_top).index.to_list()

str_result = f"On {reference_date}, the top {nb_top} products that drove the highest sales over the last {nb_months} months are: {', '.join(top_products)}."
print(str_result)
with open('results/top_five_most_sold.txt', 'w') as file:
    file.write(str_result)


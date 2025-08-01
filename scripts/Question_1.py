import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os

dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/all_stocks_5yr.csv"))
script_dir = os.path.dirname(__file__)
output_dir = os.path.abspath(os.path.join(script_dir, '..', 'picture/Question_1'))
os.makedirs(output_dir, exist_ok=True)

print(dataset.head())
print(dataset.info())
print(dataset.describe())

# Question A
# Which Companies have shown the most growth/regression over 4 years
#Calculating percent growth: 
# Last closing price - first closing price / first close

growth_df = dataset.sort_values(['Name', 'date'])
first_close = dataset.groupby('Name').first()['close']
last_close = dataset.groupby('Name').last()['close']
growth = ((last_close - first_close) / first_close) * 100
growth = growth.sort_values(ascending=False)

dataset['daily_return'] = dataset.groupby('Name')['close'].pct_change()

# Daily return standard deviation per company = volatility
volatility = dataset.groupby('Name')['daily_return'].std()
volatility.name = 'daily_volatility'

volatility_growth_df = pd.DataFrame({'growth': growth, 'daily_volatility': volatility})
volatility_growth_df.dropna(inplace=True)


#Top 5 growing compaines
print("\nTop 5 Growing Compaines")
print(growth.head())

#Top 5 regressing companies
print("\nTop 5 Regressing Companies")
print(growth.tail())

plt.figure(figsize=(10, 6))
sb.scatterplot(data=volatility_growth_df, x='growth', y='daily_volatility', hue='growth', palette='coolwarm', alpha=0.7)
plt.title('Company Growth vs. Daily Volatility (2013–2018)')
plt.xlabel('Growth %')
plt.ylabel('Daily Return Std Dev (Volatility)')
plt.axhline(volatility_growth_df['daily_volatility'].mean(), color='gray', linestyle='--', label='Avg Volatility')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "growth_vs_volatility.png"))
plt.show()


# Add group labels
volatility_growth_df['group'] = 'Middle'
volatility_growth_df.loc[growth.head(10).index, 'group'] = 'Top 10 Gainers'
volatility_growth_df.loc[growth.tail(10).index, 'group'] = 'Bottom 10 Losers'

plt.figure(figsize=(8, 6))
sb.boxplot(data=volatility_growth_df[volatility_growth_df['group'] != 'Middle'], x='group', y='daily_volatility', palette='Set2')
plt.title('Volatility Comparison: Top Gainers vs Bottom Losers')
plt.ylabel('Daily Volatility (Std Dev)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "volatility_boxplot.png"))
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os

dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/all_stocks_5yr.csv"))



dataset['date'] = pd.to_datetime(dataset['date'])

dataset = dataset.sort_values(['Name', 'date'])

grouped = dataset.groupby('Name')
returns = grouped.agg(
    first_close = ('close', 'first'),
    last_close = ('close', 'last'),
    close_var = ('close', 'var')
).reset_index()

returns['total_return'] = (returns['last_close'] - returns['first_close']) / returns['first_close']

returns['annual_return'] = (1 + returns['total_return'])**(1/4) - 1

returns['return_diff'] = (returns['annual_return'] - 0.10).abs()

closest = returns.sort_values(['return_diff', 'close_var']).head(10)

filtered = returns[(returns['annual_return'] >= 0.075) & (returns['annual_return'] <= 0.125)]

ranked = filtered.sort_values('close_var').head(10)

print("Top 10 Stocks Closest to 10% Annual Return (Sorted by Stability):")
print(closest[['Name', 'annual_return', 'close_var', 'return_diff']])

print("Top 10 Most Stable Stocks (Return between 7.5% and 12.5%):")
print(ranked[['Name', 'annual_return', 'close_var', 'return_diff']])




variances = dataset.groupby('Name')['close'].var().sort_values()
print("\nTop5 Most Stable Stocks (Lowest Variance): ")
print(variances.head())

print("\nTop 5 Most Volatile Stocks:")
print(variances.tail())

plt.figure(figsize=(10, 6))
sb.barplot(data=closest, x='annual_return', y='Name', palette='viridis')
plt.axvline(0.10, color='red', linestyle='--', label='10% target')
plt.title('Top 10 Stocks Closest to 10% Annual Return (Sorted by Stability)')
plt.xlabel('Annual Return')
plt.ylabel('Stock Name')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sb.barplot(data=ranked, x='close_var', y='Name', palette='coolwarm')
plt.title('Top 10 Most Stable Stocks (7.5% - 12.5% Annual Return)')
plt.xlabel('Close Price Variance')
plt.ylabel('Stock Name')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sb.histplot(data=returns, x='annual_return', kde=True, bins=30, color='skyblue')
plt.axvline(0.10, color='red', linestyle='--', label='10% target')
plt.title('Distribution of Annual Returns for All Stocks')
plt.xlabel('Annual Return')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sb.scatterplot(data=returns, x='annual_return', y='close_var', hue='return_diff', palette='Spectral', size='return_diff')
plt.axvline(0.10, color='red', linestyle='--', label='10% target')
plt.title('Annual Return vs Variance of Closing Price')
plt.xlabel('Annual Return')
plt.ylabel('Close Price Variance')
plt.legend()
plt.tight_layout()
plt.show()
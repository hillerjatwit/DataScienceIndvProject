import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/all_stocks_5yr.csv"))
script_dir = os.path.dirname(__file__)
output_dir = os.path.abspath(os.path.join(script_dir, '..', 'picture/Question_3'))


dataset['date'] = pd.to_datetime(dataset['date'])
dataset.sort_values(['Name', 'date'], inplace=True)

analysis_dataset = []

for name, group in tqdm(dataset.groupby('Name'), desc="Calculating Volatility and Trend"):
    stock = group.copy().sort_values('date')
    stock['volatility_30'] = stock['close'].rolling(window=30).std()
    stock['volume_avg_30'] = stock['volume'].rolling(window=30).mean()
    stock['return_30'] = stock['close'].pct_change(periods=30)
    stock['trend_30'] = stock['close'] > stock['close'].shift(30)
    stock['stock'] = name
    analysis_dataset.append(stock)

analysis_dataset = pd.concat(analysis_dataset)
analysis_dataset.dropna(subset=['volatility_30', 'volume_avg_30', 'return_30'], inplace=True)

analysis_dataset['vol_q'] = pd.qcut(analysis_dataset['volatility_30'], 4, labels=['Low Vol', 'Mid-Low Vol', 'Mid-High Vol', 'High Vol'])
analysis_dataset['volcat'] = pd.qcut(analysis_dataset['volume_avg_30'], 4, labels=['Low Vol.', 'Mid-Low Vol.', 'Mid-High Vol.', 'High Vol.'])                                

trend_summary = analysis_dataset.groupby(['vol_q', 'volcat']).agg({
    'return_30': ['mean', 'std'],
    'trend_30': 'mean'
}).reset_index()

vol_trend_summary = analysis_dataset.groupby('vol_q')['trend_30'].mean().reset_index()
best_vol = vol_trend_summary.loc[vol_trend_summary['trend_30'].idxmax()]
worst_vol = vol_trend_summary.loc[vol_trend_summary['trend_30'].idxmin()]

print("\nVolatility Quartile with Highest Upward Trend:")
print(f"{best_vol['vol_q']} - {best_vol['trend_30']:.2%}")

print("Volatility Quartile with Lowest Upward Trend:")
print(f"{worst_vol['vol_q']} - {worst_vol['trend_30']:.2%}")

# Analysis: Highest/Lowest Upward Trends by Volume Quartile
volcat_trend_summary = analysis_dataset.groupby('volcat')['trend_30'].mean().reset_index()
best_volcat = volcat_trend_summary.loc[volcat_trend_summary['trend_30'].idxmax()]
worst_volcat = volcat_trend_summary.loc[volcat_trend_summary['trend_30'].idxmin()]

print("\nVolume Quartile with Highest Upward Trend:")
print(f"{best_volcat['volcat']} - {best_volcat['trend_30']:.2%}")

print("Volume Quartile with Lowest Upward Trend:")
print(f"{worst_volcat['volcat']} - {worst_volcat['trend_30']:.2%}")

trend_summary.columns = ['Volatility Group', 'Volume Group', 'Avg 30D Return', 'Return STD', '% Upward Trend']

print("\nTrend Summary by Volatility and Volume Groups:")
print(trend_summary.sort_values('% Upward Trend', ascending=False))

plt.figure(figsize=(10, 6))
sns.boxplot(data=analysis_dataset, x='vol_q', y='return_30')
plt.title("30-Day Returns by Volatility Group")
plt.xlabel("Volatility Category")
plt.ylabel("30-Day Return")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "30-Day Returns by Volatility Group.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=analysis_dataset, x='volcat', y='return_30')
plt.title("30-Day Returns by Average Volume Group")
plt.xlabel("Volume Category")
plt.ylabel("30-Day Return")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "30-Day Returns by Average Volume Group.png"))

plt.show()

pivot_table = trend_summary.pivot(index="Volatility Group", columns="Volume Group", values="% Upward Trend")
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Probability of Upward Trend (30 Days)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Probability of Upward Trend (30 Days).png"))
plt.show()


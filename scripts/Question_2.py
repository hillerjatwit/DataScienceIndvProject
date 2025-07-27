import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# --- Load Data ---
dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/all_stocks_5yr.csv"))
dataset['date'] = pd.to_datetime(dataset['date'])

# --- Compute prev_close and target_close ---
dataset.sort_values(['Name', 'date'], inplace=True)
dataset['prev_close'] = dataset.groupby('Name')['close'].shift(1)

# --- RSI Calculation ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

all_stock_with_indicators = []
for name, group in dataset.groupby('Name'):
    stock = group.copy().sort_values('date')
    stock['ma_7'] = stock['close'].rolling(window=7).mean()
    stock['ma_30'] = stock['close'].rolling(window=30).mean()
    stock['ma_90'] = stock['close'].rolling(window=90).mean()
    stock['volatility_30'] = stock['close'].rolling(window=30).std()
    stock['momentum_10'] = stock['close'] - stock['close'].shift(10)
    stock['volume_avg_30'] = stock['volume'].rolling(window=30).mean()
    stock['rsi_14'] = compute_rsi(stock['close'], period=14)
    stock['target_close'] = stock['close'].shift(-1)
    all_stock_with_indicators.append(stock)


df = pd.concat(all_stock_with_indicators, ignore_index=True)
df.dropna(inplace=True)
basic_features = ['open', 'high', 'low', 'volume', 'prev_close']
technical_features = [
    'ma_7', 'ma_30', 'ma_90', 'volatility_30',
    'momentum_10', 'volume_avg_30', 'rsi_14'
]
all_features = basic_features + technical_features


results_basic = []
results_with_indicators = []

for stock in tqdm(df['Name'].unique(), desc="Processing Stocks"):
    stock_df = df[df['Name'] == stock].copy()
    if stock_df.shape[0] < 500:
        continue

    stock_df.sort_values('date', inplace=True)
    train_size = int(len(stock_df) * 0.6)
    train = stock_df.iloc[:train_size]
    test = stock_df.iloc[train_size:]

    y_train = train['target_close']
    y_test = test['target_close']

    X_train_basic = train[basic_features]
    X_test_basic = test[basic_features]
    scaler_basic = StandardScaler()
    X_train_basic_scaled = scaler_basic.fit_transform(X_train_basic)
    X_test_basic_scaled = scaler_basic.transform(X_test_basic)

    X_train_all = train[all_features]
    X_test_all = test[all_features]
    scaler_all = StandardScaler()
    X_train_all_scaled = scaler_all.fit_transform(X_train_all)
    X_test_all_scaled = scaler_all.transform(X_test_all)

    def evaluate_models(X_train, y_train, X_test, y_test):
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'RandomForest': RandomForestRegressor(n_estimators=100),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100)
        }
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            results[name] = rmse
        return results

    basic_rmse = evaluate_models(X_train_basic_scaled, y_train, X_test_basic_scaled, y_test)
    full_rmse = evaluate_models(X_train_all_scaled, y_train, X_test_all_scaled, y_test)

    for model in basic_rmse:
        results_basic.append({
            'Stock': stock,
            'Model': model,
            'RMSE_Basic': basic_rmse[model]
        })
        results_with_indicators.append({
            'Stock': stock,
            'Model': model,
            'RMSE_WithIndicators': full_rmse[model]
        })

basic_df = pd.DataFrame(results_basic)
full_df = pd.DataFrame(results_with_indicators)
comparison = pd.merge(basic_df, full_df, on=['Stock', 'Model'])
comparison['RMSE_Diff'] = comparison['RMSE_Basic'] - comparison['RMSE_WithIndicators']
comparison['Improved_%'] = 100 * comparison['RMSE_Diff'] / comparison['RMSE_Basic']
comparison['Improved'] = comparison['RMSE_Diff'] > 0


print("\nTop 10 Stocks Where Technical Indicators Helped Most:")
print(comparison.sort_values('Improved_%', ascending=False).head(10))

comparison.to_csv("model_comparison_with_vs_without_indicators.csv", index=False)

plt.figure(figsize=(10, 6))
sns.histplot(comparison['Improved_%'], bins=30, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title("Distribution of RMSE Improvement with Technical Indicators")
plt.xlabel("% Improvement")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=comparison, x='Model', y='Improved_%')
plt.axhline(0, color='red', linestyle='--')
plt.title("Model-Wise RMSE % Improvement Using Technical Indicators")
plt.ylabel("Improved %")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
corr = df[all_features + ['target_close']].corr()
sns.heatmap(corr[['target_close']].sort_values(by='target_close', ascending=False), annot=True, cmap='coolwarm')
plt.title("Feature Correlation with Target Close")
plt.tight_layout()
plt.show()

top_corr_features = corr['target_close'].abs().sort_values(ascending=False).index[1:4]
for feat in top_corr_features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[feat], y=df['target_close'], alpha=0.3)
    plt.title(f"Scatterplot: {feat} vs Target Close")
    plt.xlabel(feat)
    plt.ylabel("Target Close")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

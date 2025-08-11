import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# --- Load Data ---
dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/all_stocks_5yr.csv"))
dataset['date'] = pd.to_datetime(dataset['date'])

# --- Compute prev_close and target_close ---
dataset.sort_values(['Name', 'date'], inplace=True)
dataset['prev_close'] = dataset.groupby('Name')['close'].shift(1)

script_dir = os.path.dirname(__file__)
output_dir = os.path.abspath(os.path.join(script_dir, '..', 'picture/Question_2'))
os.makedirs(output_dir, exist_ok=True)


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


# --- Models with 3 sequential versions for speed ---
def get_model_versions(model_name):
    versions = []
    if model_name == 'Linear':
        # Polynomial degree 1, 2, 3
        for degree in [1, 3, 5]:
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            versions.append(model)

    elif model_name == 'Ridge':
        for alpha in [1, 0.1, 0.01]:
            versions.append(Ridge(alpha=alpha))
            
    elif model_name == 'Lasso':
        for alpha in [0.05, 0.01, 0.005]:
            versions.append(Lasso(alpha=alpha, max_iter=10000))
            
    elif model_name == 'RandomForest':
        params = [(50, 5), (100, 5), (200, 8)]
        for n_estimators, max_depth in params:
            versions.append(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42))
            
    elif model_name == 'GradientBoosting':
        params = [(50, 3), (100, 3), (200, 5)]
        for n_estimators, max_depth in params:
            versions.append(GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
    else:
        versions.append(LinearRegression())
    return versions


def evaluate_model_versions(X_train, y_train, X_test, y_test, model_name):
    versions = get_model_versions(model_name)
    results = {}
    for i, model in enumerate(versions, 1):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[f"{model_name}_v{i}"] = rmse
    return results


model_types = ['Linear', 'Ridge', 'Lasso', 'RandomForest', 'GradientBoosting']

# --- Limit stocks processed for speed ---
MAX_STOCKS = 150
stock_lengths = df.groupby('Name').size().sort_values(ascending=False)
top_stocks = stock_lengths.head(MAX_STOCKS).index.tolist()

results_versions_basic = []
results_versions_full = []

for stock in tqdm(top_stocks, desc="Processing Top Stocks"):
    stock_df = df[df['Name'] == stock].copy()
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

    for model_name in model_types:
        basic_rmse_versions = evaluate_model_versions(X_train_basic_scaled, y_train, X_test_basic_scaled, y_test, model_name)
        for version_name, rmse in basic_rmse_versions.items():
            results_versions_basic.append({
                'Stock': stock,
                'ModelVersion': version_name,
                'RMSE_Basic': rmse
            })

    for model_name in model_types:
        full_rmse_versions = evaluate_model_versions(X_train_all_scaled, y_train, X_test_all_scaled, y_test, model_name)
        for version_name, rmse in full_rmse_versions.items():
            results_versions_full.append({
                'Stock': stock,
                'ModelVersion': version_name,
                'RMSE_WithIndicators': rmse
            })

basic_versions_df = pd.DataFrame(results_versions_basic)
full_versions_df = pd.DataFrame(results_versions_full)

comparison_versions = pd.merge(basic_versions_df, full_versions_df, on=['Stock', 'ModelVersion'])
comparison_versions['RMSE_Diff'] = comparison_versions['RMSE_Basic'] - comparison_versions['RMSE_WithIndicators']
comparison_versions['Improved_%'] = 100 * comparison_versions['RMSE_Diff'] / comparison_versions['RMSE_Basic']
comparison_versions['Improved'] = comparison_versions['RMSE_Diff'] > 0

# --- Top 5 per model version ---
top5_per_model_version = comparison_versions.groupby('ModelVersion').apply(
    lambda x: x.nsmallest(5, 'RMSE_WithIndicators')
).reset_index(drop=True)

print("\nTop 5 Stocks Per Model Version:")
print(top5_per_model_version[['Stock', 'ModelVersion', 'RMSE_WithIndicators', 'Improved_%']])
top5_per_model_version.to_csv(os.path.join(output_dir, "top5_per_model_version.csv"), index=False)

# --- Plot top 5 RMSE distribution per model version ---
plt.figure(figsize=(14, 8))
sns.boxplot(data=top5_per_model_version, x='ModelVersion', y='RMSE_WithIndicators')
plt.xticks(rotation=45)
plt.title('RMSE Distribution of Top 5 Stocks Per Model Version')
plt.ylabel('RMSE (With Technical Indicators)')
plt.xlabel('Model Version')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Top5_per_model_version_RMSE_boxplot.png"))
plt.show()

plt.figure(figsize=(14, 8))
sns.stripplot(data=top5_per_model_version, x='ModelVersion', y='RMSE_WithIndicators', jitter=True, size=8, alpha=0.7)
plt.xticks(rotation=45)
plt.title('Top 5 RMSE Values Per Model Version')
plt.ylabel('RMSE (With Technical Indicators)')
plt.xlabel('Model Version')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Top5_per_model_version_RMSE_stripplot.png"))
plt.show()

# --- Top 25 models overall ---
top25_overall = comparison_versions.nsmallest(25, 'RMSE_WithIndicators')

print("\nTop 25 Best Performing Models Overall:")
print(top25_overall[['Stock', 'ModelVersion', 'RMSE_WithIndicators', 'Improved_%']])
top25_overall.to_csv(os.path.join(output_dir, "top25_best_overall.csv"), index=False)

# Extract Model base name for coloring
top25_overall['Model'] = top25_overall['ModelVersion'].str.extract(r'(^[A-Za-z]+)')

plt.figure(figsize=(14, 10))
sns.barplot(data=top25_overall, x='RMSE_WithIndicators', y='Stock', hue='Model', dodge=False)
plt.title('Top 25 Best Performing Models Overall by RMSE')
plt.xlabel('RMSE (With Technical Indicators)')
plt.ylabel('Stock')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Top25_best_overall_RMSE.png"))
plt.show()

# --- Model performance analysis ---

print("\n=== Model Performance Summary ===")
summary = comparison_versions.groupby('ModelVersion')['RMSE_WithIndicators'].agg(['mean', 'median', 'std', 'count']).reset_index()
summary.rename(columns={'mean': 'Avg_RMSE', 'median': 'Median_RMSE', 'std': 'Std_RMSE', 'count': 'Num_Stocks'}, inplace=True)

print(summary.sort_values('Avg_RMSE'))

best_model = summary.loc[summary['Avg_RMSE'].idxmin()]
print(f"\nBest Overall Model: {best_model['ModelVersion']}")
print(f"Average RMSE: {best_model['Avg_RMSE']:.4f}")
print(f"Median RMSE: {best_model['Median_RMSE']:.4f}")
print(f"Std Dev RMSE: {best_model['Std_RMSE']:.4f}")
print(f"Evaluated on {best_model['Num_Stocks']} stocks")

summary.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)


best_model_name = best_model['ModelVersion']
best_model_data = comparison_versions[comparison_versions['ModelVersion'] == best_model_name]

plt.figure(figsize=(10, 6))
sns.histplot(best_model_data['RMSE_WithIndicators'], bins=20, kde=True)
plt.title(f"RMSE Distribution of Best Model: {best_model_name}")
plt.xlabel("RMSE (With Technical Indicators)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{best_model_name}_RMSE_distribution.png"))
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(y=best_model_data['RMSE_WithIndicators'])
plt.title(f"RMSE Boxplot of Best Model: {best_model_name}")
plt.ylabel("RMSE (With Technical Indicators)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{best_model_name}_RMSE_boxplot.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=best_model_data['RMSE_Basic'],
    y=best_model_data['RMSE_WithIndicators'],
    alpha=0.7
)
plt.plot([best_model_data['RMSE_Basic'].min(), best_model_data['RMSE_Basic'].max()],
         [best_model_data['RMSE_Basic'].min(), best_model_data['RMSE_Basic'].max()],
         color='red', linestyle='--', label='No Improvement Line')
plt.title(f"Basic vs. Full RMSE for Best Model: {best_model_name}")
plt.xlabel("RMSE (Basic Features)")
plt.ylabel("RMSE (With Technical Indicators)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{best_model_name}_Basic_vs_Full_RMSE_scatter.png"))
plt.show()

from sklearn.base import is_regressor
import matplotlib.ticker as ticker

# --- Feature importance for best model ---

print(f"\nAnalyzing Feature Importance for Best Model: {best_model_name}")

# Extract model type (without version suffix)
import re
model_base = re.match(r"([A-Za-z]+)_v\d+", best_model_name).group(1)

# For feature importance, we need training data of some stock
# Let's pick the stock where this best model had lowest RMSE:
best_stock = best_model_data.loc[best_model_data['RMSE_WithIndicators'].idxmin(), 'Stock']
print(f"Using stock '{best_stock}' to fit and analyze feature importance")

stock_df = df[df['Name'] == best_stock].copy()
stock_df.sort_values('date', inplace=True)
train_size = int(len(stock_df) * 0.6)
train = stock_df.iloc[:train_size]

y_train = train['target_close']

# Use all features since best model used full features
X_train_all = train[all_features]
scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)

# Get the version number to reconstruct exact model parameters
version_number = int(best_model_name.split('_v')[1])

# Rebuild the model instance exactly:
def rebuild_model(model_base, version_number):
    if model_base == 'Linear':
        degree = [1, 2, 3][version_number - 1]
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    elif model_base == 'Ridge':
        alpha = [1, 0.1, 0.01][version_number - 1]
        model = Ridge(alpha=alpha)
    elif model_base == 'Lasso':
        alpha = [0.05, 0.01, 0.005][version_number - 1]
        model = Lasso(alpha=alpha, max_iter=10000)
    elif model_base == 'RandomForest':
        params = [(50, 5), (100, 5), (100, 8)]
        n_estimators, max_depth = params[version_number - 1]
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
    elif model_base == 'GradientBoosting':
        params = [(50, 3), (100, 3), (100, 5)]
        n_estimators, max_depth = params[version_number - 1]
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        model = LinearRegression()
    return model

best_model_instance = rebuild_model(model_base, version_number)
best_model_instance.fit(X_train_all_scaled, y_train)

# Get feature names for all_features + polynomial features if Linear
if model_base == 'Linear':
    # For PolynomialFeatures, get expanded feature names
    poly = best_model_instance.named_steps['polynomialfeatures']
    feature_names = poly.get_feature_names_out(all_features)
    coefs = best_model_instance.named_steps['linearregression'].coef_
    # Map absolute coefficient magnitude
    importance = pd.Series(np.abs(coefs), index=feature_names)
    importance = importance.sort_values(ascending=False)
else:
    # Tree-based or linear models
    if hasattr(best_model_instance, 'feature_importances_'):
        importance = pd.Series(best_model_instance.feature_importances_, index=all_features)
    elif hasattr(best_model_instance, 'coef_'):
        importance = pd.Series(np.abs(best_model_instance.coef_), index=all_features)
    else:
        importance = pd.Series(dtype=float)  # Empty if unavailable

    importance = importance.sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
importance.head(15).plot(kind='bar')
plt.title(f"Top 15 Feature Importances for Best Model: {best_model_name}")
plt.ylabel("Importance (absolute magnitude)")
plt.xlabel("Feature")
plt.grid(axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{best_model_name}_feature_importance.png"))
plt.show()
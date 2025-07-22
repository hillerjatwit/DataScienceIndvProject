import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBRegressor



# dataset = pd.read_csv("C:\Users\hillerj\DataScienceProject\DataScienceIndvProject\data\all_stocks_5yr.csv")
dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/all_stocks_5yr.csv"))
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

#Top 5 growing compaines
print("\nTop 5 Growing Compaines")
print(growth.head())

#Top 5 regressing companies
print("\nTop 5 Regressing Companies")
print(growth.tail())

# Question B: Predciting next day closing prices of a stock

# Cleaning the dataset
results = []
companies = dataset['Name'].unique()

for company in tqdm(companies, desc="Processing Companies"):
    # More Cleaning
    stock = dataset[dataset['Name'] == company].copy()
    stock['prev_close'] = stock['close'].shift(1)
    stock.dropna(inplace=True)

    if stock.shape[0] < 500:
        continue

    stock = stock.sort_values('date')
    train_size = int(len(stock) * 0.6)
    train = stock.iloc[:train_size]
    test = stock.iloc[train_size:]

    features = ['open', 'high', 'low', 'volume', 'prev_close']
    x_train = train[features]
    y_train = train['close']
    x_test = test[features]
    y_test = test['close']

    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=100),
        'XGBoost': XGBRegressor(n_estimators=100, objective='reg:squarederror')
    }

    rsme_results = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        rsme = np.sqrt(mean_squared_error(y_test, preds))
        rsme_results[name] = rsme

    best_model = min(rsme_results, key=rsme_results.get)
    best_rsme = rsme_results[best_model]

    results.append({
    'Company': company,
    'Best_Model': best_model,
    'Lowest_RMSE': round(best_rsme, 2),
    **{f'RMSE_{k}': round(v, 2) for k, v in rsme_results.items()}
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Lowest_RMSE')
print("\nTop 10 Company Models by Prediction Accuracy (including Neural Net):")
print(results_df.head(10))

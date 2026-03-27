import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'processed_jobs.csv')

df = pd.read_csv(PATH_DATA, parse_dates=['datetime'], index_col='datetime')

features = ['total_queue_gpu', 'recent_arrivals_6h', 'total_gpu_plan', 'gpu_lag_48', 'gpu_lag_1', 'day_of_week', 'hour_sin']
target = 'target_gpu_24h'

X = df[features].dropna()
y = df[target].loc[X.index]

tscv = TimeSeriesSplit(n_splits=3, test_size=144)
model = LinearRegression()

print(">>> Training Linear Baseline...")
rmses = []
for train_index, test_index in tscv.split(X):
    model.fit(X.iloc[train_index], y.iloc[train_index])
    y_pred = model.predict(X.iloc[test_index])
    rmses.append(np.sqrt(mean_squared_error(y.iloc[test_index], y_pred)))

print(f"Linear Regression Avg RMSE: {np.mean(rmses):.2f}")

print("\n>>> Feature Importance (Linear Coefficients):")
coeffs = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print(coeffs.reindex(coeffs['Coefficient'].abs().sort_values(ascending=False).index))
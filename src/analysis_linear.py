import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. LOAD & CLEAN (SAME AS BEFORE)
# ==========================================
file_path = r'C:\Users\piere\My Drive\Cours\EDHEC\M2\thesis\multivariate_cluster_series_v2.csv'
df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# FEATURES
features = [
    'total_queue_gpu',  # Demand
    'recent_arrivals_6h',  # Volatility
    'total_gpu_plan',  # Ceiling
    'gpu_lag_48',  # Baseline (Yesterday)
    'gpu_lag_1',  # Trend (Now)
    'day_of_week',
    'hour_sin'
]
target = 'target_gpu_24h'

X = df[features].dropna()
y = df[target].loc[X.index]

# ==========================================
# 2. TRAIN LINEAR REGRESSION
# ==========================================
tscv = TimeSeriesSplit(n_splits=5, test_size=144)
model = LinearRegression()

print(">>> Training Linear Regression...")
fold = 1
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Fold {fold}: RMSE = {rmse:.2f}")
    fold += 1

# ==========================================
# 3. INTERPRET THE MODEL (The "Glass Box")
# ==========================================
print("\n>>> MODEL COEFFICIENTS (How it thinks):")
coeffs = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
coeffs['Abs_Value'] = coeffs['Coefficient'].abs()
print(coeffs.sort_values('Abs_Value', ascending=False))

print(f"\nIntercept (Baseline): {model.intercept_:.2f}")

# ==========================================
# 4. FINAL BENCHMARK
# ==========================================
naive_rmse = np.sqrt(mean_squared_error(y_test, X_test['gpu_lag_48']))
print(f"\n--- RESULTS ---")
print(f"Linear Regression RMSE: {rmse:.2f}")
print(f"Naive Baseline RMSE:    {naive_rmse:.2f}")

if rmse < naive_rmse:
    print("SUCCESS: Simpler was better.")
else:
    print("FAILURE: Even Linear Regression couldn't beat the Naive guess.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# =================================================================
# 1. CONFIGURATION
# =================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'processed_jobs.csv')

TARGET_COL = 'target_gpu_24h'
TEST_SIZE_STEPS = 144  # 3 days * 48 steps/day
RANDOM_STATE = 42

# FinOps Asymmetric Cost Matrix
COST_WASTE = 1.0     # 1 unit of waste = 1x penalty
COST_SHORTAGE = 10.0 # 1 unit of downtime = 10x penalty

# =================================================================
# 2. DATA PREPARATION
# =================================================================
print(">>> Loading Target Data...")
df = pd.read_csv(PATH_DATA, parse_dates=['datetime'], index_col='datetime')

X = df.drop(columns=[TARGET_COL, 'total_gpu_plan'])
y = df[TARGET_COL]

split_point = -TEST_SIZE_STEPS
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
original_plan_test = df.iloc[split_point:]['total_gpu_plan']

# =================================================================
# 3. XGBOOST PIPELINE
# =================================================================
num_cols = [c for c in X.columns if c != 'day_of_week']
cat_cols = ['day_of_week']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=RANDOM_STATE, objective='reg:squarederror'))
])

param_distributions = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 5, 7],
    'model__subsample': [0.6, 0.8],
    'model__colsample_bytree': [0.8, 1.0]
}

tscv = TimeSeriesSplit(n_splits=3, test_size=72)

print(">>> Training Model (RandomizedSearch)...")
rs = RandomizedSearchCV(
    pipeline, param_distributions, n_iter=15, 
    scoring='neg_root_mean_squared_error', cv=tscv, random_state=RANDOM_STATE, n_jobs=-1
)
rs.fit(X_train, y_train)
y_pred_raw = rs.predict(X_test)

print(f"Base RMSE: {-rs.best_score_:.4f}")

# =================================================================
# 4. FINOPS OPTIMIZATION ALGORITHM
# =================================================================
print("\n>>> Running FinOps Buffer Optimization...")

def calculate_cost(y_true, y_pred, buffer_f, c_waste, c_short):
    provisioned = y_pred * buffer_f
    waste = np.maximum(0, provisioned - y_true)
    shortage = np.maximum(0, y_true - provisioned)
    return np.sum(waste * c_waste) + np.sum(shortage * c_short)

# Test buffers from +0% to +40%
buffer_range = np.arange(1.00, 1.40, 0.01)
costs = {b: calculate_cost(y_test.values, y_pred_raw, b, COST_WASTE, COST_SHORTAGE) for b in buffer_range}

optimal_buffer = min(costs, key=costs.get)
series_opt = pd.Series(y_pred_raw * optimal_buffer, index=y_test.index)

print(f"OPTIMAL BUFFER: {optimal_buffer:.2f}x (covers shortage risk)")

# =================================================================
# 5. BUSINESS METRICS & PLOT
# =================================================================
savings_pct = (original_plan_test.mean() - series_opt.mean()) / original_plan_test.mean() * 100
total_shortage = np.sum(np.maximum(0, y_test.values - series_opt.values))
total_waste = np.sum(np.maximum(0, series_opt.values - y_test.values))

print(f"\n[ BUSINESS RESULTS ]")
print(f"Capacity Reduction:  {savings_pct:.2f}%")
print(f"Total Shortage Risk: {total_shortage:.2f} Unit-Hours")
print(f"Total Waste Volume:  {total_waste:.2f} Unit-Hours")

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Usage', color='black', linewidth=2)
plt.plot(original_plan_test.index, original_plan_test, label='Original Static Plan', color='red', linestyle='--', alpha=0.5)
plt.plot(series_opt.index, series_opt, label=f'Optimized ML Plan ({optimal_buffer:.2f}x Buffer)', color='green', linewidth=2)

plt.fill_between(y_test.index, y_test, series_opt, where=(y_test.values > series_opt.values), color='red', alpha=0.3, label='Shortage Risk')
plt.fill_between(y_test.index, y_test, series_opt, where=(series_opt.values > y_test.values), color='green', alpha=0.1, label='Optimized Waste')

plt.title(f'Dynamic FinOps Provisioning (Cost Ratio 1:{int(COST_SHORTAGE)})', fontsize=14)
plt.ylabel('GPU Units')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'finops_result.png'))
plt.show()
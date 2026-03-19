import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor

# =================================================================
# 1. CONFIGURATION & CONSTANTS
# =================================================================
PATH_DATA = r'C:\Users\piere\My Drive\Cours\EDHEC\M2\thesis\multivariate_cluster_series_v2.csv'
TARGET_COL = 'target_gpu_24h'
TEST_SIZE_STEPS = 144  # 3 days * 48 steps/day
RANDOM_STATE = 42

# FinOps Cost Weights
COST_WASTE = 1.0  # Cost per unit of unused GPU
COST_SHORTAGE = 10.0  # Cost per unit of missing GPU (Downtime risk)

# =================================================================
# 2. DATA LOADING & SPLITTING
# =================================================================
print(">>> Loading Data...")
df = pd.read_csv(PATH_DATA, parse_dates=['datetime'], index_col='datetime')

# Feature Separation
X = df.drop([TARGET_COL, 'total_gpu_plan'], axis=1)  # Drop target & old plan
y = df[TARGET_COL]

# Time Series Split (Strict chronological split)
split_point = -TEST_SIZE_STEPS

X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
original_plan_test = df.iloc[split_point:]['total_gpu_plan']  # For comparison

print(f"Training Data: {X_train.shape}")
print(f"Testing Data:  {X_test.shape}")

# =================================================================
# 3. PIPELINE & MODEL TRAINING
# =================================================================
print(">>> Setting up Pipeline...")

num_cols = [c for c in X.columns if c != 'day_of_week']
cat_cols = ['day_of_week']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

# Model
model = XGBRegressor(random_state=RANDOM_STATE)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Hyperparameter Search
param_distributions = {
    'model__n_estimators': [100, 200, 300, 400],
    'model__learning_rate': [0.01, 0.02, 0.05],
    'model__max_depth': [3, 4, 5, 6],
    'model__subsample': [0.4, 0.6, 0.8],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__min_child_weight': [1, 5, 10]
}

tscv = TimeSeriesSplit(n_splits=5, test_size=72)

print(">>> Starting Randomized Search...")
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=50,  # Reduced slightly for speed, increase if needed
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1,
    cv=tscv,
    random_state=RANDOM_STATE
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print(f"Best RMSE: {-random_search.best_score_:.4f}")
print("Best Params:", random_search.best_params_)

# Generate Raw Predictions
y_pred_raw = best_model.predict(X_test)

# =================================================================
# 4. FINOPS OPTIMIZATION (The Strategic Shift)
# =================================================================
print("\n>>> Optimizing Provisioning Buffer...")


def calculate_total_cost(y_true, y_pred, buffer_factor, c_waste, c_short):
    """Calculates weighted cost of waste vs shortage."""
    provisioned = y_pred * buffer_factor

    waste = np.maximum(0, provisioned - y_true)
    shortage = np.maximum(0, y_true - provisioned)

    return np.sum(waste * c_waste) + np.sum(shortage * c_short)


# Optimize Buffer Factor
buffer_range = np.arange(1.00, 1.40, 0.01)
costs = {}

for buff in buffer_range:
    costs[buff] = calculate_total_cost(y_test.values, y_pred_raw, buff, COST_WASTE, COST_SHORTAGE)

optimal_buffer = min(costs, key=costs.get)
min_cost = costs[optimal_buffer]

# Create Final Optimized Plan
series_opt = pd.Series(y_pred_raw * optimal_buffer, index=y_test.index)

print(f"--------------------------------------------------")
print(f"OPTIMAL BUFFER FACTOR: {optimal_buffer:.2f}x")
print(f"   (Based on Cost Ratio -> Shortage: {COST_SHORTAGE}x vs Waste: {COST_WASTE}x)")
print(f"--------------------------------------------------")

# =================================================================
# 5. BUSINESS IMPACT EVALUATION
# =================================================================

# 1. Capacity Reduction
avg_orig = original_plan_test.mean()
avg_new = series_opt.mean()
savings_pct = (avg_orig - avg_new) / avg_orig * 100

# 2. Risk Assessment (Shortage)
total_shortage = np.sum(np.maximum(0, y_test.values - series_opt.values))
total_waste = np.sum(np.maximum(0, series_opt.values - y_test.values))

print(f"\n>>> FINAL BUSINESS METRICS")
print(f"1. Capacity Reduction:  {savings_pct:.2f}%")
print(f"2. Total Shortage Risk: {total_shortage:.2f} Unit-Hours")
print(f"3. Total Waste Volume:  {total_waste:.2f} Unit-Hours")
print(f"4. Total Optimization Score (Cost): {min_cost:.2f}")

# =================================================================
# 6. VISUALIZATION
# =================================================================
plt.figure(figsize=(15, 7))

# A. Ground Truth
plt.plot(y_test.index, y_test, label='Actual Usage', color='black', linewidth=2, alpha=0.8)

# B. Original Plan
plt.plot(original_plan_test.index, original_plan_test, label='Original Static Plan', color='red', linestyle='--',
         alpha=0.5)

# C. Optimized Plan
plt.plot(series_opt.index, series_opt, label=f'Optimized Plan ({optimal_buffer:.2f}x Buffer)', color='green',
         linewidth=2)

# D. Areas of Interest
plt.fill_between(y_test.index, y_test, series_opt,
                 where=(y_test.values > series_opt.values),
                 color='red', alpha=0.3, label='Shortage Risk (Costly)')

plt.fill_between(y_test.index, y_test, series_opt,
                 where=(series_opt.values > y_test.values),
                 color='green', alpha=0.1, label='Optimized Waste')

plt.title(f'FinOps Optimized Provisioning (Shortage Cost = {COST_SHORTAGE}x)', fontsize=16)
plt.ylabel('Total GPU Units', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


from sklearn.metrics import mean_squared_error

# 1. The "Stupid" Benchmark (Persistence)
# "Prediction for tomorrow is just today's value"
naive_pred = y_test.shift(48).fillna(method='bfill') # Assuming 48 steps = 24h lag
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))

print(f"\n--- BENCHMARK RESULTS ---")
print(f"Your Model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"Naive Model RMSE: {naive_rmse:.2f}")
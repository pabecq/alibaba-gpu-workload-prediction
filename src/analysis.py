import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np

### READ THE DATA ###
path = r'C:\Users\piere\My Drive\Cours\EDHEC\M2\thesis'
df = pd.read_csv(path + r'\multivariate_cluster_series.csv', parse_dates=['datetime'], index_col='datetime')


X = df.drop('target_gpu_24h', axis=1)
X.drop('total_gpu_plan',axis=1,  inplace=True)
y = df['target_gpu_24h']


### SPLIT THE DATA TO TRAIN TEST ###
split_point = -144

X_train = X.iloc[:split_point]
y_train = y.iloc[:split_point]

X_test = X.iloc[split_point:]
y_test = y.iloc[split_point:]

print(f"Main training set shape: {X_train.shape}")
print(f"Final holdout test set shape: {X_test.shape}")

num_cols = [c for c in X.columns if c != 'day_of_week']
cat_cols =  ['day_of_week']

num_transformer = Pipeline(steps=[(
    'imputer', SimpleImputer(strategy='median')),])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

tscv = TimeSeriesSplit(n_splits=5, test_size=72)

model = XGBRegressor(random_state=42)

param_distributions = {
    'model__n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
    'model__learning_rate': [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
    'model__max_depth': [2, 3, 4,  5, 6, 8, 10],
    'model__subsample': [0.3, 0.4, 0.5],
    'model__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    'model__gamma': [0, 0.5, 1, 1.5, 2],
    'model__reg_alpha': [0, 0.1, 0.5, 1],
    'model__reg_lambda': [0, 0.1, 0.5, 1, 2],
    'model__min_child_weight': [1, 5, 10]
}

my_pipeline= Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

random_search = RandomizedSearchCV(
    estimator=my_pipeline,
    param_distributions=param_distributions,
    n_iter=100,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1,
    cv = tscv)

random_search.fit(X_train, y_train)


print("Best parameters found:", random_search.best_params_)
print(f"Best RMSE: {-random_search.best_score_}")

# === 5. EVALUATION ===
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# FinOps Logic: 20% Buffer
recommended_request = y_pred * 1.20

# Fetch the Original Plan for the test period (from df, not X)
original_plan = df.iloc[split_point:]['total_gpu_plan']

# === 6. PLOTTING ===
# Re-index y_pred to match y_test's datetime index
series_pred = pd.Series(y_pred, index=y_test.index)
series_rec = pd.Series(recommended_request, index=y_test.index)

plt.figure(figsize=(15, 7))

# 1. Ground Truth
plt.plot(y_test.index, y_test, label='Actual GPU Usage', color='black', linewidth=2)

# 2. The Old Way (Static Plan)
plt.plot(original_plan.index, original_plan, label='Original Static Plan', color='red', linestyle='--', alpha=0.7)

# 3. The New Way (Dynamic ML Plan)
plt.plot(series_rec.index, series_rec, label='Dynamic ML Provisioning (+20% Buffer)', color='green', linestyle='-', linewidth=2)

# Areas
plt.fill_between(y_test.index, y_test, original_plan, color='red', alpha=0.1, label='Original Waste')
plt.fill_between(y_test.index, y_test, series_rec, color='green', alpha=0.1, label='Optimized Waste')

plt.title('Reducing Waste with 24h Forecasts', fontsize=16)
plt.ylabel('Total GPU Units', fontsize=12) # FIXED LABEL
plt.xlabel('Date', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate Savings
avg_orig = original_plan.mean()
avg_new = series_rec.mean()
savings_pct = (avg_orig - avg_new) / avg_orig * 100
print(f"Potential Capacity Reduction: {savings_pct:.2f}%")


def calculate_total_cost(y_true, y_pred_raw, buffer_factor, cost_waste=1.0, cost_shortage=10.0):
    """
    Calculates the total FinOps cost for a given buffer factor.

    Cost = Sum [ Cost_Waste * Max(0, Provisioned - Actual) + Cost_Shortage * Max(0, Actual - Provisioned) ]

    :param y_true: The actual GPU usage (A_t)
    :param y_pred_raw: The raw 24h GPU usage forecast from the ML model
    :param buffer_factor: The multiplier (gamma) applied to the forecast (e.g., 1.20)
    :param cost_waste: Cost weight for over-provisioning (alpha)
    :param cost_shortage: Cost weight for under-provisioning/downtime (beta)
    :return: Total calculated cost
    """
    provisioned = y_pred_raw * buffer_factor

    # 1. Calculate Waste (Over-provisioning)
    waste = np.maximum(0, provisioned - y_true)
    cost_of_waste = np.sum(waste * cost_waste)

    # 2. Calculate Shortage (Under-provisioning/Downtime)
    shortage = np.maximum(0, y_true - provisioned)
    cost_of_shortage = np.sum(shortage * cost_shortage)

    total_cost = cost_of_waste + cost_of_shortage
    return total_cost

# --- 5. EVALUATION AND OPTIMIZATION (NEW) ---
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# --- 5A. OPTIMIZE THE BUFFER ---
# Define the cost ratio: Shortage (Downtime) is 10x the cost of Waste
COST_WASTE = 1.0  # Unit cost of unused capacity
COST_SHORTAGE = 10.0 # Unit cost of downtime/shortage (Adjust this based on business risk!)

# Test a reasonable range of buffers
buffer_factors = np.arange(1.00, 1.31, 0.01)

cost_results = {}
for gamma in buffer_factors:
    cost = calculate_total_cost(y_test.values, y_pred, gamma, COST_WASTE, COST_SHORTAGE)
    cost_results[gamma] = cost

# Find the optimal buffer factor (gamma)
optimal_buffer = min(cost_results, key=cost_results.get)
min_cost = cost_results[optimal_buffer]

print(f"Optimal Buffer Factor (gamma) for Cost Ratio {COST_SHORTAGE}x: {optimal_buffer:.2f}")
print(f"Minimum Total Cost at Optimal Buffer: {min_cost:.2f}")

# The Final Dynamic ML Plan is calculated using the optimal buffer
series_opt = pd.Series(y_pred * optimal_buffer, index=y_test.index)
optimal_request = series_opt

# --- 5B. EVALUATE THE OUTCOME ---
# Calculate Savings
original_plan = df.iloc[split_point:]['total_gpu_plan']
avg_orig = original_plan.mean()
avg_opt = optimal_request.mean()
savings_pct_opt = (avg_orig - avg_opt) / avg_orig * 100

waste_opt = np.sum(np.maximum(0, optimal_request.values - y_test.values))
shortage_opt = np.sum(np.maximum(0, y_test.values - optimal_request.values))

print(f"\n--- Cost-Optimized Plan Metrics ---")
print(f"Capacity Reduction vs. Original Plan: {savings_pct_opt:.2f}%")
print(f"Total Shortage Risk: {shortage_opt:.2f} GPU-hours (Cost:{shortage_opt * COST_SHORTAGE:.2f})")
print(f"Total Waste: {waste_opt:.2f} GPU-hours (Cost:{waste_opt * COST_WASTE:.2f})")
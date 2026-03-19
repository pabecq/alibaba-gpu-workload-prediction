import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np


### READ THE DATA

path = r'C:\Users\piere\My Drive\Cours\EDHEC\M2\thesis'
df = pd.read_csv(path + r'\model_ready_data.csv', index_col='time', parse_dates=True)

X = df.drop('avg_cpu_usage', axis=1)
X.drop('cpu_waste', axis=1, inplace=True)
y = df['avg_cpu_usage']

### SPLIT THE DATA TO TRAIN TEST ###
split_point = -72

X_train = X.iloc[:split_point]
y_train = y.iloc[:split_point]

X_test = X.iloc[split_point:]
y_test = y.iloc[split_point:]

print(f"Main training set shape: {X_train.shape}")
print(f"Final holdout test set shape: {X_test.shape}")

### CREATING NAIVE 24H LAG MODEL ###


naive_24h_rmse = np.sqrt(mean_squared_error(y_test, X_test['shift_24h']))
print(f"Naive 'Lag 24h' Model RMSE: {naive_24h_rmse}")

### PREPARING XGBOOST MODEL ###

num_cols = ['cpu_request', 'shift_1h', 'shift_24h', 'rolling_avg_12h', 'rolling_avg_24h', 'rolling_max_24h']
cat_cols = ['hour', 'day_of_week', 'is_weekend']

numerical_transformer = Pipeline(steps=[(
    'imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
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

my_pipeline = Pipeline(steps=[
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


### STORE THE BEST MODEL AND PREDICT DATA ###
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)


# Create a results DataFrame
df_results = pd.DataFrame({
    'actual_usage': y_test,
    'predicted_usage': y_pred
}, index=y_test.index)

df_results.plot(figsize=(15, 6), title='Final Model: Actual vs. Predicted Usage')
plt.show()

xgboost_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Final XGBOOST RMSE: {xgboost_rmse}")
print(f"Naive 24h lag RMSE: {naive_24h_rmse}")
### COMPARISON BETWTEEN THE TWO MODELS ###

improvement = (naive_24h_rmse - xgboost_rmse) / naive_24h_rmse
print(f"Improvement {improvement}%")

### CALCULATE 95th PERCENTILE
provisioning_level = np.percentile(y_pred, 95)

print(f"Recommended CPU Request: {provisioning_level}")

##########################################
### PLOT EVERYTHING TO SHOW THE IMPACT ###
##########################################

# --- 1. Formulate Your FinOps Policy ---

# Your y_pred is the 72-hour forecast from your best_model
# Let's define our policy: 95th Percentile + 20% Safety Buffer
p95_forecast = np.percentile(y_pred, 95)
recommended_request = p95_forecast * 1.20

print(f"\n--- FinOps Policy Recommendation ---")
print(f"Model Forecasted 95th Percentile (P95): {p95_forecast:.8f}")
print(f"Recommended Provision (P95 + 20% Buffer): {recommended_request:.8f}")

# --- 2. Prepare the Plotting DataFrame ---
# This makes plotting clean and simple. We use y_test.index.
df_plot = pd.DataFrame(index=y_test.index)

# Column 1: The "Ground Truth"
df_plot['Actual Usage'] = y_test

# Column 2: The "Original Policy" (what the human set)
df_plot['Original Request'] = X_test['cpu_request']

# Column 3: Our "ML Policy" (will be a flat horizontal line)
df_plot['ML-Driven Policy (P95 + 20%)'] = recommended_request

# --- 3. Create the Final "Money Slide" Plot ---
print("Generating final policy comparison plot...")

plt.figure(figsize=(15, 7))

# Plot the "Ground Truth"
plt.plot(df_plot.index, df_plot['Actual Usage'],
         label='Actual Usage (Ground Truth)',
         color='blue',
         linewidth=1.5)

# Plot the "Original Policy"
plt.plot(df_plot.index, df_plot['Original Request'],
         label='Original Request (Human-Set)',
         color='red',
         linestyle='--',
         linewidth=2)

# Plot our "ML-Driven Policy"
plt.plot(df_plot.index, df_plot['ML-Driven Policy (P95 + 20%)'],
         label='Our ML Policy (Efficient & Safe)',
         color='green',
         linestyle='--',
         linewidth=2)

# Add FinOps context
plt.fill_between(df_plot.index,
                 df_plot['Actual Usage'],
                 df_plot['Original Request'],
                 color='red',
                 alpha=0.1,
                 label='Original Waste')

plt.fill_between(df_plot.index,
                 df_plot['Actual Usage'],
                 df_plot['ML-Driven Policy (P95 + 20%)'],
                 color='green',
                 alpha=0.1,
                 label='Provisioned Buffer')

# --- 4. Format and Save ---
plt.title('FinOps Policy Comparison: Actual vs. Recommended Provisioning', fontsize=16)
plt.ylabel('CPU Cores', fontsize=12)
plt.xlabel('Time (Final 72 Hours)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.gcf().autofmt_xdate()
plt.tight_layout()

# Save the figure for your thesis paper
plt.savefig(path + r'\finops_policy_comparison.png', dpi=300)

plt.show()



# --- 1. Formulate Your NEW Dynamic Policy ---
# y_pred is your 72-hour forecast.
# Let's create a dynamic (hourly) request based on it.
recommended_request_dynamic = y_pred * 1.20 # 20% safety buffer

print(f"\n--- Dynamic FinOps Policy ---")
print(f"This policy will apply a *different* request every hour.")

# --- 2. Prepare the Plotting DataFrame ---
df_plot = pd.DataFrame(index=y_test.index)
df_plot['Actual Usage'] = y_test
df_plot['Original Request'] = X_test['cpu_request']
# --- THIS IS THE CHANGE ---
df_plot['ML-Driven Policy (Dynamic)'] = recommended_request_dynamic

# --- 3. Create the Final "Money Slide" Plot ---
plt.figure(figsize=(15, 7))

plt.plot(df_plot.index, df_plot['Actual Usage'],
         label='Actual Usage (Ground Truth)',
         color='blue',
         linewidth=1.5)

plt.plot(df_plot.index, df_plot['Original Request'],
         label='Original Static Request (Human-Set)',
         color='red',
         linestyle='--',
         linewidth=2)

# --- THIS IS THE CHANGE ---
plt.plot(df_plot.index, df_plot['ML-Driven Policy (Dynamic)'],
         label='Our Dynamic ML Policy (Hourly)',
         color='green',
         linestyle='--',
         linewidth=2)

# --- THIS IS THE NEW WASTE CALCULATION ---
# This plot will look amazing.
plt.fill_between(df_plot.index,
                 df_plot['Actual Usage'],
                 df_plot['ML-Driven Policy (Dynamic)'],
                 color='green',
                 alpha=0.1,
                 label='New (Minimal) Waste')

# --- 4. Format and Save ---
plt.title('FinOps Policy Comparison: Static vs. Dynamic Provisioning', fontsize=16)
plt.ylabel('CPU Cores', fontsize=12)
plt.xlabel('Time (Final 72 Hours)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig(path + r'\finops_dynamic_policy.png', dpi=300)
plt.show()
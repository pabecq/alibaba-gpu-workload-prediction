import pandas as pd
import numpy as np

# =================================================================
# 1. CONFIGURATION
# =================================================================
# Update path as needed
PATH = r'C:\Users\piere\My Drive\Cours\EDHEC\M2\thesis'
OUT_PATH = PATH + r'\multivariate_cluster_series_v2.csv'

# =================================================================
# 2. DATA LOADING
# =================================================================
print(">>> 1. Loading Data Tables...")

# A. Job Table (Needed for SUBMISSION TIME to calculate Queue)
df_job = pd.read_csv(
    PATH + r'\pai_job_table.csv',
    names=['job_name', 'inst_id', 'user', 'status', 'start_time', 'end_time'],
    usecols=['job_name', 'start_time']
)
df_job.rename(columns={'start_time': 'job_submit_time'}, inplace=True)

# B. Instance Table (Execution Time)
df_instance = pd.read_csv(
    PATH + r'\pai_instance_table.csv',
    names=['job_name', 'task_name', 'inst_name', 'worker_name', 'inst_id', 'status', 'start_time', 'end_time', 'machine'],
    usecols=['job_name', 'task_name', 'worker_name', 'start_time', 'end_time']
)

# C. Sensor Table (Usage)
df_sensor = pd.read_csv(
    PATH + r'\pai_sensor_table.csv',
    names=['job_name', 'task_name', 'worker_name', 'inst_id', 'machine', 'gpu_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write', 'read_count', 'write_count'],
    usecols=['worker_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem']
)

# D. Task Table (Plan Request)
df_task = pd.read_csv(
    PATH + r'\pai_task_table.csv',
    names=['job_name', 'task_name', 'inst_num', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type'],
    usecols=['job_name', 'task_name', 'start_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type'])

# =================================================================
# 3. MERGING & PREPROCESSING
# =================================================================
print(">>> 2. Merging Datasets...")

# 1. Link Plan (Task) to Job (Submission Time)
# Required to calculate the Queue (Time between Submit and Start)
df_task_merged = pd.merge(df_task, df_job, on='job_name', how='left')

# 2. Link Sensor (Usage) to Instance (Time)
df_combined = pd.merge(df_instance, df_sensor, on='worker_name', how='inner')

# 3. Link Plan to Usage
cols_to_merge = ['job_name', 'task_name', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type']
df_combined = pd.merge(df_combined, df_task[cols_to_merge], on=['job_name', 'task_name'], how='left')

# Filter for relevant GPU jobs
df_gpu_jobs = df_combined[(df_combined['plan_gpu'] > 0) | (df_combined['gpu_wrk_util'] > 0)].copy()
df_gpu_jobs.dropna(subset=['start_time', 'end_time'], inplace=True)

print(f"    Processing {len(df_gpu_jobs)} instances for reconstruction.")

# =================================================================
# 4. CLUSTER STATE RECONSTRUCTION
# =================================================================
print(">>> 3. Reconstructing Cluster State (Usage, Plan, Queue)...")

# --- A. USAGE & PLAN STATE ---
# Event: Instance Starts -> Add Usage & Plan
events_start = pd.DataFrame({
    'time': df_gpu_jobs['start_time'],
    'd_gpu_util': df_gpu_jobs['gpu_wrk_util'].fillna(0),
    'd_gpu_plan': df_gpu_jobs['plan_gpu'].fillna(0),
    'd_cpu': df_gpu_jobs['cpu_usage'].fillna(0),
    'd_mem': df_gpu_jobs['avg_mem'].fillna(0)
})

# Event: Instance Ends -> Subtract Usage & Plan
events_end = pd.DataFrame({
    'time': df_gpu_jobs['end_time'],
    'd_gpu_util': -df_gpu_jobs['gpu_wrk_util'].fillna(0),
    'd_gpu_plan': -df_gpu_jobs['plan_gpu'].fillna(0),
    'd_cpu': -df_gpu_jobs['cpu_usage'].fillna(0),
    'd_mem': -df_gpu_jobs['avg_mem'].fillna(0)
})

# --- B. QUEUE STATE (Leading Indicator) ---
# Filter tasks that requested GPU
df_task_gpu = df_task_merged[df_task_merged['plan_gpu'] > 0].copy()

# Event: User Submits Job -> Add to Queue
queue_start = pd.DataFrame({
    'time': df_task_gpu['job_submit_time'],
    'd_queue_gpu': df_task_gpu['plan_gpu'].fillna(0)
})

# Event: Task Starts -> Remove from Queue (It becomes active)
queue_end = pd.DataFrame({
    'time': df_task_gpu['start_time'],
    'd_queue_gpu': -df_task_gpu['plan_gpu'].fillna(0)
})

# Combine all events
timeline = pd.concat([events_start, events_end, queue_start, queue_end], ignore_index=True)
timeline.sort_values('time', inplace=True)
timeline['datetime'] = pd.to_datetime(timeline['time'], unit='s')
timeline.set_index('datetime', inplace=True)

# Cumulative Sum -> State at every second
cluster_state = timeline[['d_gpu_util', 'd_gpu_plan', 'd_queue_gpu', 'd_cpu', 'd_mem']].cumsum()
cluster_state.columns = ['total_gpu_util', 'total_gpu_plan', 'total_queue_gpu', 'total_cpu_util', 'total_mem_util']
cluster_state['total_queue_gpu'] = cluster_state['total_queue_gpu'] - cluster_state['total_queue_gpu'].min()

# =================================================================
# 5. RESAMPLING & FEATURE ENGINEERING
# =================================================================
print(">>> 4. Resampling & Feature Engineering...")

# Resample to 30min intervals
df_finops = cluster_state.resample('30min').mean().ffill()

# --- JOB AGE FEATURES (Cliff Risk Indicator) ---
# Calculate "Recent Arrivals" (New work starting in last 6h)
arrivals = pd.DataFrame({
    'time': df_gpu_jobs['start_time'],
    'new_plan_gpu': df_gpu_jobs['plan_gpu'].fillna(0)
})
arrivals['datetime'] = pd.to_datetime(arrivals['time'], unit='s')
arrivals.set_index('datetime', inplace=True)

# Resample arrivals
df_arrivals = arrivals.resample('30min').sum().fillna(0)
df_finops = df_finops.join(df_arrivals)

# Recent Arrivals (Rolling 6h sum)
df_finops['recent_arrivals_6h'] = df_finops['new_plan_gpu'].rolling(window=12, min_periods=1).sum()

# Normalize Units (/100) if raw data is in % (e.g. 500 = 5 GPUs)
# Adjust this based on your specific data scale preference
df_finops['total_gpu_util'] /= 100
df_finops['total_gpu_plan'] /= 100
df_finops['total_queue_gpu'] /= 100
df_finops['total_cpu_util'] /= 100
df_finops['recent_arrivals_6h'] /= 100

# --- TARGET & LAGS ---
# Target: GPU Usage 24h later
steps_24h = 48
df_finops['target_gpu_24h'] = df_finops['total_gpu_util'].shift(-steps_24h)

# Time Features
df_finops['hour_sin'] = np.sin(2 * np.pi * df_finops.index.hour / 24)
df_finops['day_of_week'] = df_finops.index.dayofweek

# Lag Features (1 step = 30min, 48 steps = 24h)
lags = [1, 48]
for lag in lags:
    df_finops[f'gpu_lag_{lag}'] = df_finops['total_gpu_util'].shift(lag)

# Final Cleanup
df_finops.dropna(inplace=True)
df_finops.drop('time', axis=1, inplace=True)
df_finops.drop('new_plan_gpu', axis=1, inplace=True)


print(f">>> 5. Saving {len(df_finops)} rows to CSV...")
df_finops.to_csv(OUT_PATH)
print(f"Done. Saved to: {OUT_PATH}")
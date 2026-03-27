import pandas as pd
import numpy as np
from pathlib import Path
import os

# =================================================================
# 1. CONFIGURATION
# =================================================================
# Use relative paths so it works anywhere
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'raw'
OUT_PATH = ROOT / 'data' / 'processed'

# =================================================================
# 2. DATA LOADING
# =================================================================
print(">>> 1. Loading Data Tables...")

df_job = pd.read_csv(
    os.path.join(DATA_DIR, 'pai_job_table.csv'),
    names=['job_name', 'inst_id', 'user', 'status', 'start_time', 'end_time'],
    usecols=['job_name', 'start_time']
).rename(columns={'start_time': 'job_submit_time'})

df_instance = pd.read_csv(
    os.path.join(DATA_DIR, 'pai_instance_table.csv'),
    names=['job_name', 'task_name', 'inst_name', 'worker_name', 'inst_id', 'status', 'start_time', 'end_time', 'machine'],
    usecols=['job_name', 'task_name', 'worker_name', 'start_time', 'end_time']
)

df_sensor = pd.read_csv(
    os.path.join(DATA_DIR, 'pai_sensor_table.csv'),
    names=['job_name', 'task_name', 'worker_name', 'inst_id', 'machine', 'gpu_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write', 'read_count', 'write_count'],
    usecols=['worker_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem']
)

df_task = pd.read_csv(
    os.path.join(DATA_DIR, 'pai_task_table.csv'),
    names=['job_name', 'task_name', 'inst_num', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type'],
    usecols=['job_name', 'task_name', 'start_time', 'plan_cpu', 'plan_mem', 'plan_gpu']
)

# =================================================================
# 3. MERGING & RECONSTRUCTION
# =================================================================
print(">>> 2. Merging Datasets & Reconstructing State...")

df_task_merged = pd.merge(df_task, df_job, on='job_name', how='left')
df_combined = pd.merge(df_instance, df_sensor, on='worker_name', how='inner')
cols_to_merge = ['job_name', 'task_name', 'plan_cpu', 'plan_mem', 'plan_gpu']
df_combined = pd.merge(df_combined, df_task_merged[cols_to_merge], on=['job_name', 'task_name'], how='left')

df_gpu_jobs = df_combined[(df_combined['plan_gpu'] > 0) | (df_combined['gpu_wrk_util'] > 0)].copy()
df_gpu_jobs.dropna(subset=['start_time', 'end_time'], inplace=True)

# --- A. USAGE & PLAN STATE ---
events_start = pd.DataFrame({
    'time': df_gpu_jobs['start_time'],
    'd_gpu_util': df_gpu_jobs['gpu_wrk_util'].fillna(0),
    'd_gpu_plan': df_gpu_jobs['plan_gpu'].fillna(0),
    'd_cpu': df_gpu_jobs['cpu_usage'].fillna(0),
    'd_mem': df_gpu_jobs['avg_mem'].fillna(0)
})

events_end = pd.DataFrame({
    'time': df_gpu_jobs['end_time'],
    'd_gpu_util': -df_gpu_jobs['gpu_wrk_util'].fillna(0),
    'd_gpu_plan': -df_gpu_jobs['plan_gpu'].fillna(0),
    'd_cpu': -df_gpu_jobs['cpu_usage'].fillna(0),
    'd_mem': -df_gpu_jobs['avg_mem'].fillna(0)
})

# --- B. QUEUE STATE ---
df_task_gpu = df_task_merged[df_task_merged['plan_gpu'] > 0].copy()

queue_start = pd.DataFrame({
    'time': df_task_gpu['job_submit_time'],
    'd_queue_gpu': df_task_gpu['plan_gpu'].fillna(0)
})

queue_end = pd.DataFrame({
    'time': df_task_gpu['start_time'],
    'd_queue_gpu': -df_task_gpu['plan_gpu'].fillna(0)
})

timeline = pd.concat([events_start, events_end, queue_start, queue_end], ignore_index=True)
timeline.sort_values('time', inplace=True)
timeline['datetime'] = pd.to_datetime(timeline['time'], unit='s')
timeline.set_index('datetime', inplace=True)

cluster_state = timeline[['d_gpu_util', 'd_gpu_plan', 'd_queue_gpu', 'd_cpu', 'd_mem']].cumsum()
cluster_state.columns = ['total_gpu_util', 'total_gpu_plan', 'total_queue_gpu', 'total_cpu_util', 'total_mem_util']
cluster_state['total_queue_gpu'] = cluster_state['total_queue_gpu'] - cluster_state['total_queue_gpu'].min()

# =================================================================
# 4. RESAMPLING & FEATURES
# =================================================================
print(">>> 3. Resampling to 30min & Engineering Features...")

df_finops = cluster_state.resample('30min').mean().ffill()

arrivals = pd.DataFrame({'time': df_gpu_jobs['start_time'], 'new_plan_gpu': df_gpu_jobs['plan_gpu'].fillna(0)})
arrivals['datetime'] = pd.to_datetime(arrivals['time'], unit='s')
arrivals.set_index('datetime', inplace=True)
df_arrivals = arrivals.resample('30min').sum().fillna(0)
df_finops = df_finops.join(df_arrivals)

df_finops['recent_arrivals_6h'] = df_finops['new_plan_gpu'].rolling(window=12, min_periods=1).sum()

# Normalize
for col in ['total_gpu_util', 'total_gpu_plan', 'total_queue_gpu', 'total_cpu_util', 'recent_arrivals_6h']:
    df_finops[col] /= 100

# Target (T+24h) & Lags
steps_24h = 48
df_finops['target_gpu_24h'] = df_finops['total_gpu_util'].shift(-steps_24h)
df_finops['hour_sin'] = np.sin(2 * np.pi * df_finops.index.hour / 24)
df_finops['day_of_week'] = df_finops.index.dayofweek

for lag in [1, 48]:
    df_finops[f'gpu_lag_{lag}'] = df_finops['total_gpu_util'].shift(lag)

df_finops.dropna(inplace=True)
df_finops.drop(columns=['time', 'new_plan_gpu'], errors='ignore', inplace=True)

print(f">>> 4. Pipeline Complete. Saving {len(df_finops)} rows.")
df_finops.to_csv(OUT_PATH / "processed_jobs.csv")
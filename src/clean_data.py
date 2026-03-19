import pandas as pd
import numpy as np

path = r'C:\Users\piere\My Drive\Cours\EDHEC\M2\thesis'
out_path = path + r'\multivariate_cluster_series.csv'

print("1. Loading Data...")

# A. Instance Table
df_instance = pd.read_csv(
    path + r'\pai_instance_table.csv',
    names=['job_name', 'task_name', 'inst_name', 'worker_name', 'inst_id', 'status', 'start_time', 'end_time', 'machine'],
    usecols=['job_name', 'task_name', 'worker_name', 'start_time', 'end_time']
)

# B. Sensor Table
df_sensor = pd.read_csv(
    path + r'\pai_sensor_table.csv',
    names=['job_name', 'task_name', 'worker_name', 'inst_id', 'machine', 'gpu_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write', 'read_count', 'write_count'],
    usecols=['worker_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem']
)

# C. Task Table
df_task = pd.read_csv(
    path + r'\pai_task_table.csv',
    names=['job_name', 'task_name', 'inst_num', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type'],
    usecols=['job_name', 'task_name', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type']
)



#===========# MERGE DATASETS #===========#
print("2. merging hierarchies...")

# Step 1: Link Sensor (Usage) to Instance (Time) using 'worker_name'
# This gives us: [Time Start, Time End, Avg CPU, Avg GPU, Avg Mem] for every container
df_combined = pd.merge(df_instance, df_sensor, on='worker_name', how='inner')
print(f"Size of Combined DF: {len(df_combined)}")
print(f"Sample GPU Util values:\n{df_combined['gpu_wrk_util'].head()}")
# Step 2: Link Plan (Task info) to the above using 'job_name' and 'task_name'
# This gives us the "Plan" columns to compare against usage
df_combined = pd.merge(df_combined, df_task, on=['job_name', 'task_name'], how='left')

#===========# KEEP RELEVANT DATA #===========#
# Filter: Keep only relevant workloads (e.g. where GPU was requested OR used)
df_gpu_jobs = df_combined[(df_combined['plan_gpu'] > 0) | (df_combined['gpu_wrk_util'] > 0)].copy()
df_gpu_jobs.dropna(subset=['start_time', 'end_time'], inplace=True)

print(f"Processing {len(df_gpu_jobs)} instances for reconstruction...")


# ==== THE MULTIVARIATE RECONSTRUCTION ==== #
print("3. Reconstructing CPU, MEM, and GPU Curves...")

# Event: Instance Starts -> Add its average usage to the cluster total
events_start = pd.DataFrame({
    'time': df_gpu_jobs['start_time'],
    'd_gpu': df_gpu_jobs['gpu_wrk_util'].fillna(0),
    'd_cpu': df_gpu_jobs['cpu_usage'].fillna(0),       # CPU Context
    'd_mem': df_gpu_jobs['avg_mem'].fillna(0),         # Memory Context
    'd_plan_gpu': df_gpu_jobs['plan_gpu'].fillna(0)
})

# Event: Instance Ends -> Subtract its average usage
events_end = pd.DataFrame({
    'time': df_gpu_jobs['end_time'],
    'd_gpu': -df_gpu_jobs['gpu_wrk_util'].fillna(0),
    'd_cpu': -df_gpu_jobs['cpu_usage'].fillna(0),
    'd_mem': -df_gpu_jobs['avg_mem'].fillna(0),
    'd_plan_gpu': -df_gpu_jobs['plan_gpu'].fillna(0)
})

# Combine, Sort, and CumSum
timeline = pd.concat([events_start, events_end]).sort_values('time')
timeline['datetime'] = pd.to_datetime(timeline['time'], unit='s')
timeline.set_index('datetime', inplace=True)

# The Cumulative Sum creates the Cluster State at every second
cluster_state = timeline[['d_gpu', 'd_cpu', 'd_mem', 'd_plan_gpu']].cumsum()
cluster_state.columns = ['total_gpu_util', 'total_cpu_util', 'total_mem_util', 'total_gpu_plan']

# ==== RESAMPLE FOR MODEL ==== #
print("4. Resampling to 30min Intervals...")
df_finops = cluster_state.resample('30min').mean().ffill()


# ==== FEATURE ENGINEERING ==== #
print("5. Generating Features...")
# Convert GPU and CPU usage from % to units #
df_finops['total_gpu_util'] = df_finops['total_gpu_util'] /100
df_finops['total_gpu_plan'] = df_finops['total_gpu_plan'] /100
df_finops['total_cpu_util'] = df_finops['total_cpu_util'] /100


# Target: GPU Usage 24h in future
steps_24h = 48
df_finops['target_gpu_24h'] = df_finops['total_gpu_util'].shift(-steps_24h)

# Time Features
df_finops['hour_sin'] = np.sin(2 * np.pi * df_finops.index.hour / 24)
df_finops['day_of_week'] = df_finops.index.dayofweek

# Lag Features (Context from the past)
# Now we include CPU and MEM history as predictors for GPU!
lags = [1, 48] # 30min ago, 24h ago
for lag in lags:
    df_finops[f'gpu_lag_{lag}'] = df_finops['total_gpu_util'].shift(lag)
    df_finops[f'cpu_lag_{lag}'] = df_finops['total_cpu_util'].shift(lag) # CPU context
    df_finops[f'mem_lag_{lag}'] = df_finops['total_mem_util'].shift(lag) # Mem context

df_finops.dropna(inplace=True)

start_row = 144
df_clean = df_finops.iloc[start_row:].copy()

print("Saving...")
df_clean.to_csv(out_path)
print(f"Saved. You now have {len(df_clean)} rows with CPU, GPU, and MEM history.")
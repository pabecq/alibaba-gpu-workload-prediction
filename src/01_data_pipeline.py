import pandas as pd
import numpy as np
import os

# =================================================================
# 1. CONFIGURATION
# =================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed')

print(">>> 1. Chargement des Tables...")

df_job = pd.read_csv(os.path.join(DATA_DIR, 'pai_job_table.csv'), header=None,
    names=['job_name', 'inst_id', 'user', 'status', 'start_time', 'end_time'],
    usecols=['job_name', 'start_time']).rename(columns={'start_time': 'job_submit_time'})

df_instance = pd.read_csv(os.path.join(DATA_DIR, 'pai_instance_table.csv'), header=None,
    names=['job_name', 'task_name', 'inst_name', 'worker_name', 'inst_id', 'status', 'start_time', 'end_time', 'machine'],
    usecols=['job_name', 'task_name', 'worker_name', 'start_time', 'end_time', 'machine'])

df_sensor = pd.read_csv(os.path.join(DATA_DIR, 'pai_sensor_table.csv'), header=None,
    names=['job_name', 'task_name', 'worker_name', 'inst_id', 'machine', 'gpu_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write', 'read_count', 'write_count'],
    usecols=['worker_name', 'cpu_usage', 'gpu_wrk_util'])

df_task = pd.read_csv(os.path.join(DATA_DIR, 'pai_task_table.csv'), header=None,
    names=['job_name', 'task_name', 'inst_num', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type'],
    usecols=['job_name', 'task_name', 'start_time', 'plan_gpu'])

df_spec = pd.read_csv(os.path.join(DATA_DIR, 'pai_machine_spec.csv'), header=None,
    names=['machine', 'gpu_type', 'cap_cpu', 'cap_mem', 'cap_gpu'])
TOTAL_CLUSTER_CAPACITY = df_spec['cap_gpu'].sum()
print(f"    [INFO] Capacité Totale Physique du Datacenter : {TOTAL_CLUSTER_CAPACITY} GPUs")

# =================================================================
# 2. RECONSTRUCTION ÉVÉNEMENTIELLE (Le Moteur Core)
# =================================================================
print(">>> 2. Fusion et Reconstruction de l'état (Timeline)...")

df_task_merged = pd.merge(df_task, df_job, on='job_name', how='left')
df_combined = pd.merge(df_instance, df_sensor, on='worker_name', how='inner')
df_combined = pd.merge(df_combined, df_task_merged[['job_name', 'task_name', 'plan_gpu']], on=['job_name', 'task_name'], how='left')

df_gpu_jobs = df_combined[(df_combined['plan_gpu'] > 0) | (df_combined['gpu_wrk_util'] > 0)].copy()
df_gpu_jobs.dropna(subset=['start_time', 'end_time'], inplace=True)

# Événements d'Utilisation et de Réservation
events_start = pd.DataFrame({'time': df_gpu_jobs['start_time'], 'd_gpu_util': df_gpu_jobs['gpu_wrk_util'].fillna(0), 'd_gpu_plan': df_gpu_jobs['plan_gpu'].fillna(0)})
events_end = pd.DataFrame({'time': df_gpu_jobs['end_time'], 'd_gpu_util': -df_gpu_jobs['gpu_wrk_util'].fillna(0), 'd_gpu_plan': -df_gpu_jobs['plan_gpu'].fillna(0)})

# Événements de File d'attente (Queue)
df_task_gpu = df_task_merged[df_task_merged['plan_gpu'] > 0].copy()
queue_start = pd.DataFrame({'time': df_task_gpu['job_submit_time'], 'd_queue': df_task_gpu['plan_gpu'].fillna(0)})
queue_end = pd.DataFrame({'time': df_task_gpu['start_time'], 'd_queue': -df_task_gpu['plan_gpu'].fillna(0)})

timeline = pd.concat([events_start, events_end, queue_start, queue_end], ignore_index=True).sort_values('time')

# CORRECTION 2 : Fuseau horaire (UTC vers Asia/Shanghai)
timeline['datetime'] = pd.to_datetime(timeline['time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
timeline.set_index('datetime', inplace=True)

# L'intégrale mathématique : la charge à la seconde près
cluster_state = timeline[['d_gpu_util', 'd_gpu_plan', 'd_queue']].cumsum()
cluster_state.columns = ['gpu_util', 'gpu_plan', 'gpu_queue']

# Normalisation (Alibaba donne souvent des % où 100 = 1 GPU)
cluster_state /= 100 

# =================================================================
# 3. LE SECRET FINOPS : AGRÉGATION PAR LE PIRE SCÉNARIO (MAX)
# =================================================================
print(">>> 3. Resampling FinOps (Extraction des Micro-bursts)...")

df_resampled = cluster_state.resample('30min').agg({
    'gpu_util': ['mean', 'max'],
    'gpu_plan': ['mean'],
    'gpu_queue': ['mean']
}).ffill()

# Aplatir les colonnes MultiIndex créées par agg()
df_resampled.columns = ['util_mean', 'util_max', 'plan_mean', 'queue_mean']

# Remplacement de la taille absolue par la dynamique (la dérivée)
df_resampled['queue_trend'] = df_resampled['queue_mean'].diff().fillna(0)

# On supprime la colonne absolue qui est fausse
df_resampled.drop(columns=['queue_mean'], inplace=True)

df_resampled['cluster_capacity'] = TOTAL_CLUSTER_CAPACITY

steps_24h = 48
df_resampled['target_util_max_24h'] = df_resampled['util_max'].shift(-steps_24h)

# Ces features sont maintenant parfaitement alignées avec le cycle circadien chinois
fractional_hour = df_resampled.index.hour + (df_resampled.index.minute / 60.0)
df_resampled['hour_sin'] = np.sin(2 * np.pi * fractional_hour / 24)
df_resampled['hour_cos'] = np.cos(2 * np.pi * fractional_hour / 24)
df_resampled['day_of_week'] = df_resampled.index.dayofweek

df_resampled.dropna(inplace=True)

warmup_threshold = TOTAL_CLUSTER_CAPACITY * 0.075
steady_state_mask = df_resampled['util_mean'] > warmup_threshold

# On trouve l'index du premier moment où la condition est respectée
if steady_state_mask.any():
    first_valid_idx = steady_state_mask.idxmax()
    df_resampled = df_resampled.loc[first_valid_idx:].copy()
else:
    print("[ATTENTION] Le cluster n'a jamais atteint 5% de charge !")

# --- CORRECTION DU WARM-DOWN (RIGHT CENSORING) ---
# On sait que l'écrasante majorité des jobs durent moins de 3 jours.
# Les 3 derniers jours du dataset sont donc corrompus car les jobs longs 
# n'ont pas de date de fin enregistrée et ont été supprimés de l'analyse.
warm_down_threshold = 3 * 48

df_resampled = df_resampled.iloc[:-warm_down_threshold].copy()

print(f">>> 4. Pipeline Terminé. {len(df_resampled)} séquences générées (Warmup purgé).")

if os.path.exists(OUT_PATH):
    df_resampled.to_csv(os.path.join(OUT_PATH, 'processed_jobs.csv'))
else: 
    os.mkdir(OUT_PATH)
    df_resampled.to_csv(os.path.join(OUT_PATH, 'processed_jobs.csv'))

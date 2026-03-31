import pandas as pd

def load_and_preprocess(filepath, target_gpu='V100'):
    print(f"\n>>> Chargement des données pour le GPU : {target_gpu}...")
    df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)
    
    # Le filtre métier absolu
    df_filtered = df[df['gpu_type'] == target_gpu].copy()
    
    if df_filtered.empty:
        raise ValueError(f"Aucune donnée trouvée pour le GPU {target_gpu}")

    features = ['util_mean', 'util_max', 'plan_mean', 'queue_trend', 'lag_util_max_24h', 'hour_sin', 'hour_cos', 'day_of_week']
    target = 'target_util_max_24h'

    X = df_filtered[features]
    y = df_filtered[target]

    split_idx = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"    [INFO] Train set : {len(X_train)} points | Test set : {len(X_test)} points.")

    return X_train, X_test, y_train, y_test
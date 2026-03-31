import os
import pandas as pd


def preprocess(df):
    features = [
    'util_mean', 'util_max', 'plan_mean', 'queue_trend', 
    'lag_util_max_24h', 'hour_sin', 'hour_cos', 'day_of_week'
    ]
    target = 'target_util_max_24h'

    X = df[features]
    y = df[target]

    # Séparation Train/Test strictement chronologique (On garde les 20 derniers % pour le test)
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"    [INFO] Set d'entraînement : {len(X_train)} points.")
    print(f"    [INFO] Set de test : {len(X_test)} points.")

    return X, y, X_train, X_test, y_train, y_test
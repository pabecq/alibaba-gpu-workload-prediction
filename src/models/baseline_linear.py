import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_utils import preprocess

# =================================================================
# 1. CONFIGURATION ET CHARGEMENT
# =================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_jobs.csv')
OUT_PATH = os.path.join(BASE_DIR, 'outputs')

df = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)

# =================================================================
# 2. PRÉPARATION DES MATRICES (Features & Target)
# =================================================================

X, y, X_train, X_test, y_train, y_test = preprocess(df)

# =================================================================
# 3. PIPELINE DE MODÉLISATION (Régression Linéaire)
# =================================================================
def baseline_linear(X_train, X_test, y_train, y_test):
    print("\n>>> Entraînement du modèle de référence (Baseline)...")

    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', QuantileRegressor(quantile=0.95, alpha=0.0, solver='highs')) 
    ])

    baseline_pipeline.fit(X_train, y_train)
    y_pred = baseline_pipeline.predict(X_test)

    return y_pred

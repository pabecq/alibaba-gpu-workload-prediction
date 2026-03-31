import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def run_xgboost_model(X_train, y_train, X_test):
    print("    [RUN] Entraînement du Challenger (XGBoost Quantile 0.95)...")

    # Utilisation de tree_method='hist' pour accélérer drastiquement l'entraînement
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.95,
            n_estimators=200,      # Nombre d'arbres
            learning_rate=0.05,    # Vitesse d'apprentissage
            max_depth=5,           # Profondeur pour capter les interactions complexes
            tree_method='hist',    # Optimisation mémoire et CPU
            random_state=42
        ))
    ])

    xgb_pipeline.fit(X_train, y_train)
    y_pred = xgb_pipeline.predict(X_test)

    return y_pred
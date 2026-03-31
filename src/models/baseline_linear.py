from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor

def run_baseline_linear(X_train, y_train, X_test):
    print("    [RUN] Entraînement de la Baseline (Linear Quantile 0.95)...")

    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', QuantileRegressor(quantile=0.95, alpha=0.0, solver='highs')) 
    ])

    baseline_pipeline.fit(X_train, y_train)
    y_pred = baseline_pipeline.predict(X_test)

    return y_pred
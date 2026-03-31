import os
from data_utils import load_and_preprocess
from models.baseline_linear import run_baseline_linear
from models.xgboost_model import run_xgboost_model
from evaluate import evaluate_finops, plot_finops_forecast

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_jobs.csv')

GPU_PRICES = {
    'V100': 10.0,
    'P100': 5.0,
    'T4': 2.0,
    'MISC': 1.0
}

def main():
    print(">>> DÉMARRAGE DE L'ÉVALUATION GLOBALE DU DATACENTER <<<")
    
    total_datacenter_cost = 0.0
    
    for gpu_type, price in GPU_PRICES.items():
        print(f"\n{'='*50}")
        print(f"--- TRAITEMENT DU PARC : {gpu_type} (Prix unitaire: {price}$) ---")
        
        try:
            # 1. Extraction
            X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH, target_gpu=gpu_type)
            
            # --- MODÈLE 1 : BASELINE (LINÉAIRE) ---
            y_pred_linear = run_baseline_linear(X_train, y_train, X_test)
            cost_lin, _, _ = evaluate_finops(y_test, y_pred_linear, model_name="Linear Quantile", quantile=0.95, unit_price=price)
            plot_finops_forecast(y_test, y_pred_linear, 
                                 model_title="Linear Quantile 0.95", 
                                 file_prefix="baseline", 
                                 gpu_type=gpu_type)
            
            # --- MODÈLE 2 : CHALLENGER (XGBOOST) ---
            y_pred_xgb = run_xgboost_model(X_train, y_train, X_test)
            cost_xgb, _, _ = evaluate_finops(y_test, y_pred_xgb, model_name="XGBoost Quantile", quantile=0.95, unit_price=price)
            plot_finops_forecast(y_test, y_pred_xgb, 
                                 model_title="XGBoost Quantile 0.95", 
                                 file_prefix="xgboost",   
                                 gpu_type=gpu_type)
            
            # On ajoute SEULEMENT le coût du meilleur modèle (XGBoost) au total pour voir l'économie
            total_datacenter_cost += min(cost_xgb, cost_lin)
            
        except Exception as e:
            print(f"[ERREUR] Impossible de traiter {gpu_type} : {e}")

    print(f"\n{'='*50}")
    print(f"COÛT TOTAL DU DATACENTER (BASELINE) : {total_datacenter_cost:,.2f} $")

if __name__ == "__main__":
    main()
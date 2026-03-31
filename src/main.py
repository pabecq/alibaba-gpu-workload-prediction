import os
from data_utils import load_and_preprocess
from models.baseline_linear import run_baseline_linear
# from evaluate import evaluate_finops 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_jobs.csv')

# Le dictionnaire des prix (Le cœur de ton FinOps)
# Exemple arbitraire : Un V100 coûte 10x plus cher à louer qu'un vieux MISC.
GPU_PRICES = {
    'V100': 10.0,
    'P100': 5.0,
    'T4': 2.0,
    'MISC': 1.0
}

def main():
    print(">>> DÉMARRAGE DE L'ÉVALUATION GLOBALE DU DATACENTER <<<")
    
    total_datacenter_cost = 0.0
    
    # On boucle sur chaque type de matériel
    for gpu_type, price in GPU_PRICES.items():
        print(f"\n{'='*50}")
        print(f"--- TRAITEMENT DU PARC : {gpu_type} (Prix unitaire: {price}$) ---")
        
        try:
            # 1. Chargement spécifique au GPU
            X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH, target_gpu=gpu_type)
            
            # 2. Entraînement de la Baseline aveugle
            y_pred_linear = run_baseline_linear(X_train, y_train, X_test)
            
            # 3. Évaluation Financière (À brancher avec ton evaluate.py plus tard)
            # cost, waste, shortage = evaluate_finops(y_test, y_pred_linear, model_name=f"Linear {gpu_type}", quantile=0.95, unit_price=price)
            # total_datacenter_cost += cost
            
            print(f"[SUCCÈS] Modèle Baseline entraîné pour {gpu_type}.")
            
        except Exception as e:
            print(f"[ERREUR] Impossible de traiter {gpu_type} : {e}")

    print(f"\n{'='*50}")
    # print(f"COÛT TOTAL DU DATACENTER (BASELINE) : {total_datacenter_cost:,.2f} $")

if __name__ == "__main__":
    main()
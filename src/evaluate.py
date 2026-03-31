import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def evaluate_finops(y_true, y_pred, model_name="Model", quantile=0.95, unit_price=1.0):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    shortages = np.maximum(0, y_true - y_pred)
    wastes = np.maximum(0, y_pred - y_true)
    
    cost_waste = unit_price 
    cost_shortage = unit_price * (quantile / (1.0 - quantile))
    
    total_shortage = shortages.sum()
    total_waste = wastes.sum()
    total_cost = (total_waste * cost_waste) + (total_shortage * cost_shortage)
    
    print(f"    -> Gaspillage (Waste) : {total_waste:.1f} GPUs à {cost_waste}$/u")
    print(f"    -> Pénurie (Shortage) : {total_shortage:.1f} GPUs (Pénalité x{cost_shortage/unit_price:.0f})")
    print(f"    => COÛT FINANCIER: {total_cost:,.2f} $")
    
    return total_cost, total_waste, total_shortage

def plot_finops_forecast(y_true, y_pred, model_title="Model", file_prefix="modele", gpu_type="V100", window=240):
    """
    model_title : Le beau titre qui s'affiche SUR l'image (ex: "XGBoost Quantile 0.95")
    file_prefix : Le nom technique fixe du fichier (ex: "xgboost") pour garantir l'écrasement.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'outputs', 'img') # Ajout du sous-dossier img
    os.makedirs(save_dir, exist_ok=True)

    y_true_zoom = y_true.iloc[-window:] if hasattr(y_true, 'iloc') else y_true[-window:]
    y_pred_zoom = y_pred[-window:]
    
    if hasattr(y_true, 'index'):
        x_axis = y_true.index[-window:]
    else:
        x_axis = np.arange(len(y_true_zoom))

    plt.figure(figsize=(15, 6))
    
    plt.plot(x_axis, y_true_zoom, label='Vérité Terrain (Pic Réel)', color='black', linewidth=2)
    plt.plot(x_axis, y_pred_zoom, label=f'Prédiction', color='blue', linestyle='--')

    plt.fill_between(x_axis, y_true_zoom, y_pred_zoom, 
                     where=(np.array(y_true_zoom) > np.array(y_pred_zoom)), 
                     color='red', alpha=0.3, label='DANGER : Pénurie (Shortage)')

    # On utilise le beau titre pour l'affichage
    plt.title(f'Stratégie FinOps: {model_title} face aux Pics de Charge [{gpu_type}]', fontsize=14, fontweight='bold')
    plt.ylabel(f'Cores GPU {gpu_type}')
    plt.xlabel('Temps')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # On utilise le préfixe strict pour le nom de fichier (Écrasement garanti)
    filename = f"{file_prefix}_{gpu_type.lower()}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    print(f"    [GRAPHIQUE] Écrasé/Sauvegardé sous : {filepath}")
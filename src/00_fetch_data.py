import os
import urllib.request
import tarfile
import sys

# =================================================================
# 1. CONFIGURATION
# =================================================================
# On remonte d'un cran (hors de src/) pour cibler le dossier data/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Les liens officiels d'Alibaba OSS
DATASETS = {
    "pai_job_table": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_job_table.tar.gz",
    "pai_instance_table": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_instance_table.tar.gz",
    "pai_sensor_table": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_sensor_table.tar.gz",
    "pai_task_table": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz",
    "pai_machine_spec": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_spec.tar.gz" # Ajouté pour le plafond capacitaire !
}

# =================================================================
# 2. FONCTIONS UTILITAIRES
# =================================================================
def reporthook(count, block_size, total_size):
    """Affiche une barre de progression simple dans le terminal"""
    progress_size = int(count * block_size)
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\r... Téléchargement : {percent}% ({progress_size / (1024*1024):.1f} MB)")
    sys.stdout.flush()

def download_and_extract(name, url, dest_dir):
    csv_file = os.path.join(dest_dir, f"{name}.csv")
    tar_file = os.path.join(dest_dir, f"{name}.tar.gz")

    # Vérification intelligente
    if os.path.exists(csv_file):
        print(f"[OK] {name}.csv est déjà présent. Ignoré.")
        return

    print(f"\n[DOWNLOAD] Récupération de {name} depuis Alibaba OSS...")
    try:
        urllib.request.urlretrieve(url, tar_file, reporthook)
        print(f"\n[EXTRACT] Décompression de {name}...")
        
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=dest_dir)
            
        # Nettoyage : on supprime l'archive pour ne pas doubler l'espace disque
        os.remove(tar_file)
        print(f"[SUCCESS] {name} prêt.")
        
    except Exception as e:
        print(f"\n[ERREUR] Problème avec {name}: {e}")

# =================================================================
# 3. EXÉCUTION
# =================================================================
if __name__ == "__main__":
    print(">>> DÉMARRAGE DU DATA FETCHER FINOPS <<<")
    
    # Création du dossier data/ s'il n'existe pas
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Dossier créé : {DATA_DIR}")

    # Boucle sur les datasets
    for name, url in DATASETS.items():
        download_and_extract(name, url, DATA_DIR)

    print("\n>>> TOUTES LES DONNÉES SONT PRÊTES <<<")
    print("Tu peux maintenant lancer src/01_data_pipeline.py")
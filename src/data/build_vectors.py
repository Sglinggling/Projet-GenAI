import sys
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration des chemins pour pouvoir importer les modules internes
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.data.load_data import load_raw_csv
from src.data.preprocess import preprocess_movies

def main():
    print("--- 1. Chargement des données ---")
    csv_path = ROOT_DIR / "src" / "data" / "raw" / "imdb_movies.csv"
    
    # On charge et on nettoie les données
    df_raw = load_raw_csv(str(csv_path))
    df = preprocess_movies(df_raw)
    
    print("--- 2. Chargement du modèle SBERT ---")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"--- 3. Calcul des vecteurs en cours")
    # On utilise semantic_text de process.py
    embeddings = model.encode(df["semantic_text"].tolist(), show_progress_bar=True)
    
    output_path = ROOT_DIR / "src" / "data" / "processed" / "movies_with_embeddings.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"--- 4. Sauvegarde dans {output_path} ---")
    with open(output_path, "wb") as fOut:
        pickle.dump({'df': df, 'embeddings': embeddings}, fOut)
        
    print("Les vecteurs sont prêts.")

if __name__ == "__main__":
    main()
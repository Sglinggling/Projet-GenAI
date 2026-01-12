from src.config import configure_hf_ssl_bypass
configure_hf_ssl_bypass()

from src.data.load_data import load_raw_csv
from src.data.preprocess import preprocess_movies
from src.nlp.embedder import load_embedder
from src.recommender.recommend import recommend_top_k, compute_scores
from src.genai.client import generate_text


def main():
    # ---- Data ----
    df_raw = load_raw_csv("data/raw/imdb_movies.csv")
    df = preprocess_movies(df_raw)

    # ---- Model ----
    model = load_embedder()

    # ---- User input (test) ----
    user_text = "I want a dark psychological thriller with intense atmosphere, mystery, and twists."

    # ---- Reco ----
    top3 = recommend_top_k(df, model, user_text, k=3)

    print(top3[["Series_Title","Released_Year","Genre","IMDB_Rating","semantic_score"]])

    # ---- Validation scores (optionnel) ----
    df_scored = compute_scores(df, model, user_text)

    print("\nTOP 5 pertinents")
    print(df_scored.sort_values("semantic_score", ascending=False)
          [["Series_Title","Genre","semantic_score"]].head(5))

    print("\nTOP 5 hors sujet")
    print(df_scored.sort_values("semantic_score", ascending=True)
          [["Series_Title","Genre","semantic_score"]].head(5))

    # ---- GenAI (Ollama) ----
    prompt = f"""
Tu es un assistant cinéma. Réponds en français.

Profil utilisateur :
{user_text}

Top 3 films recommandés :
1) {top3.iloc[0]['Series_Title']} – {top3.iloc[0]['Overview']}
2) {top3.iloc[1]['Series_Title']} – {top3.iloc[1]['Overview']}
3) {top3.iloc[2]['Series_Title']} – {top3.iloc[2]['Overview']}

Explique pourquoi ces films correspondent au profil (3–5 phrases chacun),
puis dresse un court profil cinéphile de l’utilisateur (2–3 phrases).
"""

    print("\n--- GenAI (Ollama) ---\n")
    print(generate_text(prompt))


if __name__ == "__main__":
    main()

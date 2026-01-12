from src.config import configure_hf_ssl_bypass
configure_hf_ssl_bypass()

from src.data.load_data import load_raw_csv
from src.data.preprocess import preprocess_movies
from src.nlp.embedder import load_embedder
from src.recommender.recommend import recommend_top_k, compute_scores


def main():
    df_raw = load_raw_csv("data/raw/imdb_movies.csv")
    df = preprocess_movies(df_raw)

    model = load_embedder()

    user_text = "I want a dark psychological thriller with intense atmosphere, mystery, and twists."

    # Top 3 recommandations
    top3 = recommend_top_k(df, model, user_text, k=3)
    print(top3[["Series_Title","Released_Year","Genre","IMDB_Rating","semantic_score"]])

    # --- TEST / VALIDATION DU SCORE ---
    df_scored = compute_scores(df, model, user_text)

    print("\nTOP 5 pertinents")
    print(df_scored.sort_values("semantic_score", ascending=False)
          [["Series_Title","Genre","semantic_score"]].head(5))

    print("\nTOP 5 hors sujet")
    print(df_scored.sort_values("semantic_score", ascending=True)
          [["Series_Title","Genre","semantic_score"]].head(5))


if __name__ == "__main__":
    main()

from src.data.load_data import load_raw_csv
from src.data.preprocess import preprocess_movies
from src.nlp.embedder import load_embedder
from src.recommender.recommend import recommend_top_k

def main():
    df_raw = load_raw_csv("data/raw/imdb_movies.csv")
    df = preprocess_movies(df_raw)

    model = load_embedder()

    user_text = "I want a dark psychological thriller with intense atmosphere, mystery, and twists."
    top3 = recommend_top_k(df, model, user_text, k=3)

    print(top3[["Series_Title","Released_Year","Genre","IMDB_Rating","semantic_score"]])

if __name__ == "__main__":
    main()

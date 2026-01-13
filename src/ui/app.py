import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import streamlit as st

from src.config import configure_hf_ssl_bypass
configure_hf_ssl_bypass()

from src.data.load_data import load_raw_csv
from src.data.preprocess import preprocess_movies
from src.nlp.embedder import load_embedder
from src.recommender.recommend import recommend_top_k
from src.genai.client import generate_text


@st.cache_resource
def get_model():
    return load_embedder()

@st.cache_data
def load_movies():
    df_raw = load_raw_csv("data/raw/imdb_movies.csv")
    return preprocess_movies(df_raw)


def main():
    st.set_page_config(page_title="Movie Reco (SBERT + Ollama)", layout="wide")
    st.title("🎬 Recommandation de films")

    df = load_movies()
    model = get_model()

    user_text = st.text_area(
        "Décris le film que tu veux (ambiance, émotions, style…) :",
        value="Je veux un thriller psychologique sombre avec du mystère et des rebondissements."
    )

    if st.button("Recommander"):
        top3 = recommend_top_k(df, model, user_text, k=3)

        st.subheader("🏆 Top 3 recommandations")
        st.dataframe(
            top3[["Series_Title", "Released_Year", "Genre", "IMDB_Rating", "semantic_score", "Overview"]],
            use_container_width=True,
            hide_index=True
        )

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

        st.subheader("Explication GenAI")
        st.write(generate_text(prompt))


if __name__ == "__main__":
    main()

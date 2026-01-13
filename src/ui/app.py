import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.config import configure_hf_ssl_bypass
configure_hf_ssl_bypass()

from src.data.load_data import load_raw_csv
from src.data.preprocess import preprocess_movies
from src.nlp.embedder import load_embedder
from src.recommender.recommend import compute_scores
from src.genai.client import generate_text


@st.cache_resource
def get_model():
    return load_embedder()


@st.cache_data
def load_movies():
    df_raw = load_raw_csv("data/raw/imdb_movies.csv")
    df = preprocess_movies(df_raw)
    df["year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def build_query_text(free_text: str, mood: str, preferred_genres: list[str]) -> str:
    preferred_txt = ", ".join(preferred_genres) if preferred_genres else "no specific genre"
    return f"{free_text.strip()}. Mood: {mood}. Preferred genres: {preferred_txt}."


def build_user_profile(
    free_text: str, mood: str, year_min: int, year_max: int, preferred_genres: list[str]
) -> str:
    return (
        f"Texte libre: {free_text}\n"
        f"Mood: {mood}\n"
        f"Période: {year_min}-{year_max}\n"
        f"Genres préférés: {', '.join(preferred_genres) if preferred_genres else 'non spécifié'}"
    )


def build_genai_prompt(user_profile: str, top3: pd.DataFrame) -> str:
    films_txt = "\n".join(
        [
            f"{rank}. {row['Series_Title']} ({row['Released_Year']}), Genre: {row['Genre']}. Synopsis: {row['Overview']}"
            for rank, (_, row) in enumerate(top3.iterrows(), start=1)
        ]
    )

    return f"""
Tu es un assistant cinéma. Réponds en français.

IMPORTANT :
- Structure ta réponse EXACTEMENT comme suit.
- Ne répète aucune section.
- N’ajoute pas de section "Raisonnement".

## Profil cinéphile de l'utilisateur
(2–3 phrases)

## Recommandations de films
{films_txt}

Pour chaque film (1, 2, 3), explique en 3–5 phrases pourquoi il correspond au profil utilisateur.

Termine par un court paragraphe de synthèse (2 phrases maximum).
""".strip()



def inject_css():
    st.markdown(
        """
<style>
/* --- Global background --- */
.stApp {
  background: linear-gradient(180deg, #f5f9ff 0%, #ffffff 60%);
}

/* --- Main layout --- */
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

/* --- Titles --- */
h1 {
  font-size: 2.1rem !important;
  letter-spacing: -0.02em;
}
.section-title {
  font-size: 1.35rem;
  font-weight: 900;
  margin-top: 0.3rem;
  margin-bottom: 0.6rem;
  color: #0f172a;
}

/* --- Sidebar polish --- */
section[data-testid="stSidebar"] {
  background: #f8fbff;
  border-right: 1px solid #e6eef8;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
  color: #0f172a;
}

/* --- Cards --- */
.movie-card {
  position: relative;
  padding: 16px;
  border: 1px solid #e2ecf8;
  border-radius: 18px;
  background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  height: 100%;
}
.movie-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 18px 38px rgba(15, 23, 42, 0.14);
  border-color: #c9dcf5;
}

.rank-pill {
  position:absolute;
  top:12px; right:12px;
  font-size:12px;
  font-weight: 900;
  color:#0b3b66;
  background: #e9f3ff;
  border:1px solid #cfe4ff;
  border-radius:999px;
  padding:5px 10px;
}

.chip {
  display:inline-block;
  font-size:12px;
  font-weight:800;
  padding:4px 10px;
  border-radius:999px;
  border:1px solid #e2ecf8;
  background:#f3f7ff;
  color:#0f172a;
  margin-right:6px;
  margin-top:6px;
}

.muted { color:#64748b; }

/* --- Justification block (anchor-based) --- */
#genai-box + div {
  background: rgba(255,255,255,0.92);
  border: 1px solid #e6eef8;
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
}

/* Titres markdown à l'intérieur de la box */
#genai-box + div h2 {
  font-size: 1.25rem !important;
  margin: 0.6rem 0 0.4rem 0 !important;
  letter-spacing: -0.01em;
  color: #0f172a;
}
#genai-box + div h3 {
  font-size: 1.05rem !important;
  margin: 0.5rem 0 0.25rem 0 !important;
  color: #0f172a;
}

/* Texte */
#genai-box + div p,
#genai-box + div li {
  font-size: 1.03rem;
  line-height: 1.6;
  color: #1f2937;
}
#genai-box + div ol,
#genai-box + div ul {
  margin-top: 0.4rem !important;
}

/* --- Fix global markdown titles (GenAI output) --- */
div[data-testid="stMarkdownContainer"] h2 {
  font-size: 1.25rem !important;
  margin: 0.8rem 0 0.4rem 0 !important;
  letter-spacing: -0.01em;
  color: #0f172a;
}

div[data-testid="stMarkdownContainer"] h3 {
  font-size: 1.1rem !important;
  margin: 0.6rem 0 0.3rem 0 !important;
  color: #0f172a;
}

</style>
""",
        unsafe_allow_html=True,
    )



def render_movie_card(row: pd.Series, rank: int):
    title = row["Series_Title"]
    year = row["Released_Year"]
    genre = row["Genre"]
    rating = row["IMDB_Rating"]
    overview = row["Overview"]
    director = row.get("Director", "")

    st.markdown(
        f"""
<div class="movie-card">
  <div class="rank-pill">#{rank}</div>

  <div style="font-size:18px;font-weight:950;margin-bottom:2px;color:#0f172a;">
    {title} <span class="muted" style="font-weight:750;">({year})</span>
  </div>

  <div class="muted" style="font-size:13px;margin-bottom:6px;">
    By <b>{director if director else "—"}</b>
  </div>

  <div>
    <span class="chip">🎭 {genre}</span>
    <span class="chip">⭐ IMDB {rating}</span>
  </div>

  <div style="margin-top:10px;color:#334155;line-height:1.55;">
    {overview}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )



def main():
    st.set_page_config(page_title="🎬 CineMatch", layout="wide")
    inject_css()

    st.title("🎬 CineMatch")

    df = load_movies()
    model = get_model()

    # -------- Sidebar --------
    st.sidebar.header("Profil cinéphile")

    free_text = st.sidebar.text_area(
        "Décris le film que tu veux (ambiance, émotions, style...)",
        value="Je veux un thriller psychologique avec du suspense.",
    )

    mood = st.sidebar.selectbox(
        "Mood recherché",
        ["Suspense", "Drôle", "Triste", "Romantique", "Sombre", "Psychologique", "Inspirant", "Intense"],
    )

    year_min, year_max = st.sidebar.slider(
        "Période de sortie",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=(1995, int(df["year"].max())),
    )

    st.sidebar.subheader("Genres préférés")
    common_genres = [
        "Comedy",
        "Drama",
        "Romance",
        "Thriller",
        "Action",
        "Horror",
        "Sci-Fi",
        "Mystery",
        "Crime",
        "Adventure",
    ]
    preferred_genres = st.sidebar.multiselect(
        "Sélectionne un ou plusieurs genres",
        options=common_genres,
        default=["Thriller"] if "Thriller" in common_genres else [],
    )

    run_btn = st.sidebar.button("🔎 Recommander", use_container_width=True)

    # -------- Main --------
    if not run_btn:
        st.info("Complète le profil à gauche puis clique sur **Recommander**.")
        return

    # Filtre période
    df_filtered = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()
    if len(df_filtered) < 50:
        df_filtered = df.copy()

    # Filtre léger par genres (si choisis)
    if preferred_genres:
        mask = df_filtered["Genre"].apply(lambda g: any(pg.lower() in str(g).lower() for pg in preferred_genres))
        df_genre = df_filtered[mask].copy()
        if len(df_genre) >= 30:
            df_filtered = df_genre

    query_text = build_query_text(free_text, mood, preferred_genres)
    user_profile = build_user_profile(free_text, mood, year_min, year_max, preferred_genres)

    df_scored = compute_scores(df_filtered, model, query_text)
    top3 = df_scored.sort_values("semantic_score", ascending=False).head(3)

    st.caption(
        f"Résultats pour : **{mood}** · **{year_min}-{year_max}** · Genres : **{', '.join(preferred_genres) or 'Tous'}**"
    )

    # ✅ Top3
    st.markdown('<div class="section-title">🏆 Top 3 recommandations</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows(), start=1):
        with cols[i - 1]:
            render_movie_card(row, i)

    st.markdown("<hr class='soft-hr'/>", unsafe_allow_html=True)

    # ✅ Justification & Profil (plus user friendly + texte plus gros)
    st.markdown('<div class="section-title">Justification & Profil</div>', unsafe_allow_html=True)
    prompt = build_genai_prompt(user_profile, top3)
    gen_text = generate_text(prompt)
    with st.container():
        st.markdown('<div class="explain-wrap">', unsafe_allow_html=True)
        st.markdown(gen_text)
        st.markdown('</div>', unsafe_allow_html=True)


    # ✅ Détails prof en bas
    st.divider()
    st.caption("Détails techniques")

    with st.expander("Comment ça marche"):
        st.write(
            "L’application encode la requête utilisateur et les descriptions narratives des films avec SBERT, "
            "puis calcule une similarité cosinus pour obtenir un score d’affinité. "
            "Les 3 meilleurs films sont proposés. La GenAI (Ollama) génère ensuite une justification et un profil cinéphile."
        )

    with st.expander("Voir la requête sémantique envoyée à SBERT"):
        st.code(query_text)

    with st.expander("Voir le profil utilisateur structuré"):
        st.text(user_profile)

    with st.expander("Voir les scores sémantiques"):
        st.dataframe(
            top3[["Series_Title", "semantic_score"]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()

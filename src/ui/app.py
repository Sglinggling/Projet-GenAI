import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sentence_transformers import util
import markdown
 
# Configuration des chemins
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
 
from src.config import configure_hf_ssl_bypass
configure_hf_ssl_bypass()
 
from src.data.load_data import load_raw_csv
from src.data.preprocess import preprocess_movies
from src.nlp.embedder import load_embedder
from src.recommender.recommend import compute_scores
from src.genai.client import generate_text
 
from src.genai.prompt import build_genai_prompt

@st.cache_resource
def get_model():
    return load_embedder()
 
 
@st.cache_data
def load_movies():
    csv_path = ROOT_DIR / "src" / "data" / "raw" / "imdb_movies.csv"
    df_raw = load_raw_csv(str(csv_path))
    df = preprocess_movies(df_raw)
   
    df["year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
    df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")
    df = df.dropna(subset=["year", "IMDB_Rating"]).copy()
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
 
def load_css_file(css_file_path):
    with open(css_file_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
 
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
  <div class="movie-title">{title} <span style="color:#888; font-size:0.8em; font-weight:normal;">({year})</span></div>
  <div style="font-size:13px; color:#aaa; margin-bottom:10px;">De {director}</div>
  <div style="margin-bottom:15px;">
    <span class="chip">{genre}</span>
    <span class="chip-rating">★ {rating}</span>
  </div>
  <div style="font-size:14px; color:#bbb; line-height:1.5;">{overview[:200]}...</div>
</div>
""",
        unsafe_allow_html=True,
    )
 
 
def analyze_semantic_breakdown(top3_df, model, free_text, mood, preferred_genres):
    genre_text = ", ".join(preferred_genres) if preferred_genres else "General"
    emb_genre = model.encode(genre_text, convert_to_tensor=True)
    emb_mood = model.encode(mood, convert_to_tensor=True)
    emb_text = model.encode(free_text, convert_to_tensor=True)
   
    movie_texts = top3_df["semantic_text"].tolist()
    emb_movies = model.encode(movie_texts, convert_to_tensor=True)
   
    scores_genre = util.cos_sim(emb_genre, emb_movies)[0].cpu().numpy()
    scores_mood = util.cos_sim(emb_mood, emb_movies)[0].cpu().numpy()
    scores_text = util.cos_sim(emb_text, emb_movies)[0].cpu().numpy()
   
    data = []
    for i, (idx, row) in enumerate(top3_df.iterrows()):
        data.append({"Film": f"#{i+1} {row['Series_Title'][:12]}..", "Critère": "Genre", "Score": int(scores_genre[i] * 100)})
        data.append({"Film": f"#{i+1} {row['Series_Title'][:12]}..", "Critère": "Ambiance", "Score": int(scores_mood[i] * 100)})
        data.append({"Film": f"#{i+1} {row['Series_Title'][:12]}..", "Critère": "Histoire", "Score": int(scores_text[i] * 100)})
       
    return pd.DataFrame(data)
 
 
def render_target_chart(top3_df):
    st.markdown('<div class="section-title">Carte de Proximité</div>', unsafe_allow_html=True)
    st.caption("Visualisation de la distance sémantique. Plus c'est au centre, plus c'est parfait.")
 
    max_score = top3_df['semantic_score'].max()
    min_score = top3_df['semantic_score'].min()
    score_range = max_score - min_score
    if score_range == 0: score_range = 1
 
    fig = go.Figure()
 
    # --- CERCLES EN BLANC (Visibilité accrue) ---
    rings = [0.2, 0.4, 0.6]
    labels = ["Top Match", "Très proche", "Pertinent"]
    for r, label in zip(rings, labels):
        fig.add_shape(type="circle", xref="x", yref="y", x0=-r, y0=-r, x1=r, y1=r,
            line=dict(color="white", width=1, dash="dot"), opacity=0.4, fillcolor="rgba(0,0,0,0)")
        fig.add_annotation(x=0, y=r, text=label, showarrow=False,
            font=dict(size=12, color="#ddd"), yshift=8)
 
    # Centre
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
        marker=dict(size=20, color='#ffffff', symbol='cross-thin', line=dict(width=3)),
        hoverinfo='none'))
    fig.add_annotation(x=0, y=0, text="VOUS", showarrow=False, yshift=-20, font=dict(color="white", size=12, family="Oswald"))
 
    angles = [90, 210, 330]
    for i, (idx, row) in enumerate(top3_df.iterrows()):
        score = row['semantic_score']
        visual_radius = 0.15 + ((max_score - score) / score_range * 0.40) if score_range > 0.001 else 0.35
       
        theta_rad = np.radians(angles[i])
        x = visual_radius * np.cos(theta_rad)
        y = visual_radius * np.sin(theta_rad)
 
        color = '#e50914' if i == 0 else ('#b20710' if i == 1 else '#82050b')
 
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=50, color=color, line=dict(width=3, color='white')), # TAILLE AUGMENTÉE
            text=[f"#{i+1}"], textposition="middle center",
            textfont=dict(color='white', size=20, family='Arial Black'),
            name=f"#{i+1} {row['Series_Title']}",
            hovertemplate=f"<b>{row['Series_Title']}</b><br>Score: {score:.4f}<extra></extra>"
        ))
 
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.7, 0.7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.7, 0.7], scaleanchor="x", scaleratio=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        height=650, # GRANDE TAILLE DEMANDÉE
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
 
 
def render_breakdown_chart(breakdown_df):
    st.markdown('<div class="section-title">Pourquoi ces films ?</div>', unsafe_allow_html=True)
   
    fig = go.Figure()
    colors = {"Genre": "#e50914", "Ambiance": "#b20710", "Histoire": "#555555"}
   
    for critere in ["Genre", "Ambiance", "Histoire"]:
        subset = breakdown_df[breakdown_df["Critère"] == critere]
        fig.add_trace(go.Bar(
            y=subset["Film"], x=subset["Score"],
            name=critere, orientation='h', marker_color=colors[critere],
            text=subset["Score"].apply(lambda x: f"{x}%"), textposition='auto',
        ))
 
    fig.update_layout(
        barmode='group',
        xaxis=dict(range=[0, 110], showgrid=True, gridcolor='#333', zeroline=False, title="Correspondance (%)", title_font=dict(color="#888")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#ccc")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Lato'),
        legend=dict(orientation="h", y=1.1, font=dict(color="#ccc")),
        margin=dict(l=10, r=10, t=10, b=10),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
 
 
def main():
    st.set_page_config(page_title="CineMatch", layout="wide", page_icon="🍿")
   
    css_path = ROOT_DIR / "assets" / "style.css"
    if css_path.exists():
        load_css_file(css_path)
 
    df = load_movies()
    model = get_model()
 
    # --- SIDEBAR ---
    st.sidebar.header("CRITÈRES")
    # Plus de height fixée ici, le CSS s'en charge (min-height: 250px)
    free_text = st.sidebar.text_area("Envie du moment", value="I want a dark psychological thriller...")
    mood = st.sidebar.selectbox("Ambiance", ["Suspense", "Drôle", "Triste", "Romantique", "Sombre", "Psychologique", "Inspirant", "Intense"])
    year_min, year_max = st.sidebar.slider("Époque", 1920, 2024, (1990, 2024))
    preferred_genres = st.sidebar.multiselect("Genres", ["Thriller", "Documentary", "Fantasy", "Mystery", "Family" "Drama", "Sci-Fi", "Action", "Horror"], ["Thriller"])
   
    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("LANCER LA RECHERCHE", use_container_width=True)
 
    # --- HEADER ---
    st.markdown("""
        <div class="logo-container">
            <div class="logo-text">
                <span class="logo-white">CINE</span><span class="logo-red">MATCH</span>
            </div>
            <div class="logo-subtitle">Votre assistant cinéma propulsé par l'IA</div>
        </div>
    """, unsafe_allow_html=True)
 
    if not run_btn:
        return
 
    # --- LOGIC ---
    df_filtered = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()
    if preferred_genres:
        mask = df_filtered["Genre"].apply(lambda g: any(pg.lower() in str(g).lower() for pg in preferred_genres))
        if mask.sum() > 0: df_filtered = df_filtered[mask]
 
    query_text = build_query_text(free_text, mood, preferred_genres)
    user_profile = build_user_profile(free_text, mood, year_min, year_max, preferred_genres)
 
    df_scored = compute_scores(df_filtered, model, query_text)
    top3 = df_scored.sort_values("semantic_score", ascending=False).head(3)
    breakdown_df = analyze_semantic_breakdown(top3, model, free_text, mood, preferred_genres)
 
    # --- 1. AFFICHE ---
    st.markdown('<div class="section-title">À l\'affiche pour vous</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    for i, (_, row) in enumerate(top3.iterrows(), start=1):
        with cols[i - 1]:
            render_movie_card(row, i)
 
    # --- 2. IA EXPERT ---
    st.markdown('<div class="section-title">Un Avis d\'expert</div>', unsafe_allow_html=True)
    prompt = build_genai_prompt(user_profile, top3)
    with st.spinner("Analyse en cours..."):
        gen_text = generate_text(prompt)
   
    html_text = markdown.markdown(gen_text)
    st.markdown(f'<div class="explain-wrap">{html_text}</div>', unsafe_allow_html=True)
 
    # --- 3. DATAVIZ (ORDRE INVERSÉ + TARGET GÉANTE) ---
    render_breakdown_chart(breakdown_df)  # Barres d'abord
    render_target_chart(top3)             # Cible ensuite (Géante)
   
    # --- 4. DETAILS TECHNIQUES ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.caption("Détails techniques")
 
    with st.expander("Comment ça marche"):
        st.write("L'application encode la requête utilisateur et les descriptions narratives dans les films avec SBERT, puis calcule une similarité cosinus pour obtenir un score d'affinité. Les trois meilleurs films sont proposés. La GenAI génère ensuite une justification et un profil cinéphile.")
 
    with st.expander("Voir la requête sémantique"):
        st.code(query_text)
 
    with st.expander("Scores détaillés"):
        st.dataframe(top3[["Series_Title", "semantic_score"]], use_container_width=True)
 
if __name__ == "__main__":
    main()
 
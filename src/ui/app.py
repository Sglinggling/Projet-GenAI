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
    
    # --- NOUVEAU : Calcul du pourcentage ---
    raw_score = row.get("semantic_score", 0)
    match_percentage = int(raw_score * 100)

    st.markdown(
        f"""
<div class="movie-card">
  <div class="rank-pill">#{rank}</div>
  <div class="movie-title">{title} <span style="color:#888; font-size:0.8em; font-weight:normal;">({year})</span></div>
  <div style="font-size:13px; color:#aaa; margin-bottom:10px;">De {director}</div>
  
  <div style="margin-bottom:15px; display: flex; align-items: center; gap: 10px;">
    <span class="match-score">{match_percentage}% Match</span>
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
    st.markdown('<div class="section-title">Radar de Pertinence</div>', unsafe_allow_html=True)
    st.caption("Plus le film est proche du centre, plus il correspond à votre recherche.")

    fig = go.Figure()

    # --- 1. LE FOND ---
    rings = [0.3, 0.6, 0.9]
    for r in rings:
        fig.add_shape(type="circle", xref="x", yref="y", x0=-r, y0=-r, x1=r, y1=r,
            line=dict(color="#333", width=1), layer="below")
    
    # Croix centrale
    fig.add_trace(go.Scatter(x=[-1.2, 1.2], y=[0, 0], mode="lines", line=dict(color="#222", width=1), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=[0, 0], y=[-1.2, 1.2], mode="lines", line=dict(color="#222", width=1), hoverinfo="none"))

    # --- 2. LE CENTRE (VOUS) ---
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=14, color='#ffffff', symbol='x', line=dict(width=2)),
        text=["<b>VOUS</b>"], textposition="bottom center",
        textfont=dict(color="#888", size=12),
        hoverinfo='none'
    ))

    # --- 3. LES FILMS ---
    angles_deg = [90, 330, 210]
    radii = [0.4, 0.7, 0.8] 

    for i, (idx, row) in enumerate(top3_df.iterrows()):
        angle = np.radians(angles_deg[i])
        r = radii[i]
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        color = '#e50914' if i == 0 else ('#b20710' if i == 1 else '#82050b')
        
        # Calcul du pourcentage
        match_score = int(row['semantic_score'] * 100)

        # A. BULLE (Chiffre)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=35, color=color, line=dict(width=2, color='white')),
            text=[str(i+1)], textposition="middle center",
            textfont=dict(color='white', size=18, family="Oswald", weight="bold"),
            hoverinfo='text',
            hovertext=f"{row['Series_Title']}"
        ))

        # B. ÉTIQUETTE (Titre + Score)
        # Positionnement intelligent pour éviter les collisions
        if i == 0: # HAUT
            x_anc, y_anc = "center", "bottom"
            ay_offset = -28
            ax_offset = 0
        elif i == 1: # DROITE
            x_anc, y_anc = "left", "middle"
            ay_offset = 0
            ax_offset = 28
        else: # GAUCHE
            x_anc, y_anc = "right", "middle"
            ay_offset = 0
            ax_offset = -28

        # On ajoute le score en vert dans le texte
        label_text = f"<b>{row['Series_Title']}</b> <span style='color:#46d369; font-size:12px'>({match_score}%)</span>"

        fig.add_annotation(
            x=x, y=y,
            text=label_text,
            showarrow=True,
            arrowhead=0, arrowwidth=1, arrowcolor="#555",
            ax=ax_offset, ay=ay_offset,
            xanchor=x_anc, yanchor=y_anc,
            bgcolor="rgba(10, 10, 10, 0.8)", # Fond sombre
            bordercolor=color, borderwidth=1, borderpad=6,
            font=dict(size=14, color="#eee", family="Lato")
        )

    # --- 4. LAYOUT ZOOMÉ ---
    fig.update_layout(
        # Zoom ajusté ici (plus le chiffre est petit, plus c'est zoomé)
        xaxis=dict(range=[-1.05, 1.05], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.05, 1.05], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        height=650
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
 
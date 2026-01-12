import re
import pandas as pd

def clean_text(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def build_semantic_text(row: pd.Series) -> str:
    title = clean_text(row.get("Series_Title", ""))
    year = clean_text(row.get("Released_Year", ""))
    genre = clean_text(row.get("Genre", ""))
    overview = clean_text(row.get("Overview", ""))
    director = clean_text(row.get("Director", ""))
    stars = ", ".join([clean_text(row.get("Star1","")), clean_text(row.get("Star2","")), clean_text(row.get("Star3",""))]).strip(", ").strip()

    # Texte composite (référentiel narratif + metadata)
    return (
        f"Title: {title}. "
        f"Year: {year}. "
        f"Genre: {genre}. "
        f"Plot: {overview}. "
        f"Director: {director}. "
        f"Stars: {stars}."
    ).strip()

def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["Series_Title","Released_Year","Certificate","Runtime","Genre",
              "IMDB_Rating","Overview","Director","Star1","Star2","Star3"]
    df = df[needed].copy()

    # Nettoyage
    for col in needed:
        df[col] = df[col].apply(clean_text)

    # ID
    df["FilmID"] = range(1, len(df) + 1)

    # Semantic text (pour SBERT)
    df["semantic_text"] = df.apply(build_semantic_text, axis=1)

    # Optionnel: tokens keywords simples (utile pour affichage)
    df["Keywords"] = (df["Genre"] + "; " + df["Director"] + "; " +
                      df["Star1"] + "; " + df["Star2"] + "; " + df["Star3"]).str.strip("; ").str.strip()

    return df

# src/genai/prompts.py
import pandas as pd

def build_genai_prompt(user_profile: str, top3: pd.DataFrame) -> str:
    """
    Construit le prompt système pour l'expert cinéma.
    """
    films_txt = "\n".join(
        [
            f"FILM_ID_{rank}: {row['Series_Title']} ({row['Released_Year']}), Genre: {row['Genre']}. Synopsis: {row['Overview']}"
            for rank, (_, row) in enumerate(top3.iterrows(), start=1)
        ]
    )
 
    return f"""
Tu es un critique cinéma expert pour "CineMatch".
Analyse le profil et recommande ces 3 films.
 
PROFIL :
{user_profile}
 
FILMS :
{films_txt}
 
FORMAT (MARKDOWN) :
1. Titre H1 (#) : "Analyse du Profil". (2 phrases max)
2. Titre H1 (#) : "La Sélection".
3. Pour chaque film, Titre H3 (###) : Titre du film (Année).
4. Sous chaque film : Un paragraphe explicatif clair et lisible.
5. Pas de numérotation automatique.
 
Ton : Expert, élégant, direct.
""".strip()
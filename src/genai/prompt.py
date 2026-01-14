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
Tu es un expert critique de cinéma. Ton but est de recommander ces films de manière percutante.

PROFIL UTILISATEUR :
{user_profile}

FILMS SELECTIONNÉS :
{films_txt}

CONSIGNES DE RÉPONSE (RESPECTE STRICTEMENT CE FORMAT MARKDOWN) :
Ne mets pas d'introduction (pas de "Voici ma sélection"). Commence direct.

# VOTRE PROFIL
(Ici, une analyse rapide du profil cinéphile en 2 phrases).

# NOTRE SÉLECTION

### {top3.iloc[0]['Series_Title']} ({top3.iloc[0]['Released_Year']})
(Paragraphe vendeur expliquant pourquoi ce film matche le profil).

### {top3.iloc[1]['Series_Title']} ({top3.iloc[1]['Released_Year']})
(Paragraphe vendeur...).

### {top3.iloc[2]['Series_Title']} ({top3.iloc[2]['Released_Year']})
(Paragraphe vendeur...).

Ton : Expert, élégant, direct. Pas de listes à puces.
""".strip()
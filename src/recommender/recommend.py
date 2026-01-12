import numpy as np
import pandas as pd
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

def recommend_top_k(
    df: pd.DataFrame,
    model: SentenceTransformer,
    user_text: str,
    k: int = 3
) -> pd.DataFrame:
    movie_texts = df["semantic_text"].tolist()

    user_emb = model.encode([user_text], convert_to_tensor=True)
    movie_emb = model.encode(movie_texts, convert_to_tensor=True)

    scores = util.cos_sim(user_emb, movie_emb)[0].cpu().numpy()
    out = df.copy()
    out["semantic_score"] = scores

    return out.sort_values("semantic_score", ascending=False).head(k)

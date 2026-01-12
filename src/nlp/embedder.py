from sentence_transformers import SentenceTransformer

def load_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)

import os
import json
import requests
from src.genai.cache import cache_get, cache_set

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

def generate_text(prompt: str) -> str:
    cached = cache_get(prompt)
    if cached is not None:
        return cached

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    text = r.json().get("response", "").strip()

    cache_set(prompt, text)
    return text

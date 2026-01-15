import hashlib
import json
from pathlib import Path

CACHE_PATH = Path("data/processed/genai_cache.json")

def _key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def cache_get(prompt: str):
    if not CACHE_PATH.exists():
        return None
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return data.get(_key(prompt))

def cache_set(prompt: str, response: str):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if CACHE_PATH.exists():
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    data[_key(prompt)] = response
    CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

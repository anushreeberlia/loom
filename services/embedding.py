import os
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"


def build_embedding_text(item: dict) -> str:
    """Build deterministic text string for embedding from BaseItem."""
    parts = []
    
    if item.get("category"):
        parts.append(f"Category: {item['category']}")
    
    if item.get("primary_color"):
        parts.append(f"Color: {item['primary_color']}")
    
    if item.get("fit") and item["fit"] != "unknown":
        parts.append(f"Fit: {item['fit']}")
    
    if item.get("material"):
        parts.append(f"Material: {item['material']}")
    
    if item.get("style_tags"):
        parts.append(f"Style: {', '.join(item['style_tags'])}")
    
    if item.get("occasion_tags"):
        parts.append(f"Occasion: {', '.join(item['occasion_tags'])}")
    
    if item.get("season_tags"):
        parts.append(f"Season: {', '.join(item['season_tags'])}")
    
    return ". ".join(parts) + "." if parts else ""


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from OpenAI."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=30.0
    )
    
    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.text}")
    
    data = response.json()
    return data["data"][0]["embedding"]


def embed_base_item(base_item: dict) -> list[float]:
    """Generate embedding for a BaseItem dict."""
    text = build_embedding_text(base_item)
    return get_embedding(text)


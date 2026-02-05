import os
import psycopg2
import httpx
import time
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = "postgresql://localhost:5432/outfit_styler"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# OpenAI text-embedding-3-small outputs 1536 dimensions
EMBEDDING_MODEL = "text-embedding-3-small"


def build_embedding_text(item: dict) -> str:
    """Build deterministic text string for embedding."""
    parts = [
        item["name"],
        f"Category: {item['category']}",
        f"Color: {item['primary_color']}",
    ]
    
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
    
    return ". ".join(parts) + "."


def get_embedding(text: str) -> list[float]:
    """Call OpenAI embedding API."""
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
        raise Exception(f"API error: {response.text}")
    
    data = response.json()
    return data["data"][0]["embedding"]


def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Get a key at: https://platform.openai.com/api-keys")
        print("Add to .env: OPENAI_API_KEY=sk-...")
        return
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Get items without embeddings (tagged OR untagged with basic info)
    cursor.execute("""
        SELECT id, name, category, primary_color, fit, material,
               style_tags, occasion_tags, season_tags
        FROM catalog_items 
        WHERE embedding IS NULL
          AND name IS NOT NULL
          AND category IS NOT NULL;
    """)
    
    rows = cursor.fetchall()
    columns = ["id", "name", "category", "primary_color", "fit", "material",
               "style_tags", "occasion_tags", "season_tags"]
    
    items = [dict(zip(columns, row)) for row in rows]
    
    total = len(items)
    print(f"Found {total} items needing embeddings")
    
    if total == 0:
        print("Nothing to embed!")
        cursor.close()
        conn.close()
        return
    
    embedded = 0
    failed = 0
    
    for i, item in enumerate(items):
        item_id = item["id"]
        name = item["name"]
        
        print(f"[{i+1}/{total}] [{item_id}] {name[:50]}...", end=" ", flush=True)
        
        try:
            text = build_embedding_text(item)
            embedding = get_embedding(text)
            
            # Store as pgvector format
            cursor.execute(
                "UPDATE catalog_items SET embedding = %s WHERE id = %s",
                (embedding, item_id)
            )
            conn.commit()
            print("✓")
            embedded += 1
            
        except Exception as e:
            print(f"✗ {str(e)[:60]}")
            failed += 1
        
        time.sleep(0.1)  # OpenAI has high rate limits
    
    cursor.close()
    conn.close()
    print(f"\n{'='*50}")
    print(f"DONE! Embedded: {embedded}, Failed: {failed}")


if __name__ == "__main__":
    main()

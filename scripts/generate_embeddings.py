"""
Generate FashionCLIP embeddings for catalog items.

FashionCLIP 2.0 (512-dim) replaces OpenAI text-embedding-3-small (1536-dim).
For catalog items with images, embeds the image directly.
For items without images, falls back to text embedding.

Run: python scripts/generate_embeddings.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import httpx
import time
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")

from services.fashion_clip import embed_text, embed_image
from services.embedding import build_embedding_text


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, category, primary_color, fit, material,
               style_tags, occasion_tags, season_tags, image_url
        FROM catalog_items
        WHERE embedding IS NULL
          AND name IS NOT NULL
          AND category IS NOT NULL;
    """)

    rows = cursor.fetchall()
    columns = ["id", "name", "category", "primary_color", "fit", "material",
               "style_tags", "occasion_tags", "season_tags", "image_url"]

    items = [dict(zip(columns, row)) for row in rows]

    total = len(items)
    print(f"Found {total} items needing FashionCLIP embeddings")

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
            image_url = item.get("image_url")
            embedding = None

            # Prefer image embedding (direct CLIP encoding)
            if image_url:
                try:
                    resp = httpx.get(image_url, timeout=15.0, follow_redirects=True)
                    if resp.status_code == 200:
                        embedding = embed_image(resp.content)
                        print("(image)", end=" ")
                except Exception:
                    pass

            # Fallback to text embedding
            if embedding is None:
                text = build_embedding_text(item)
                embedding = embed_text(text)
                print("(text)", end=" ")

            cursor.execute(
                "UPDATE catalog_items SET embedding = %s WHERE id = %s",
                (embedding, item_id)
            )
            conn.commit()
            print("ok")
            embedded += 1

        except Exception as e:
            print(f"FAIL {str(e)[:60]}")
            failed += 1

    cursor.close()
    conn.close()
    print(f"\n{'='*50}")
    print(f"DONE! Embedded: {embedded}, Failed: {failed}")


if __name__ == "__main__":
    main()

"""
Migration script: OpenAI text-embedding-3-small (1536-dim) → FashionCLIP 2.0 (512-dim)

This script:
1. Alters all vector columns from vector(1536) to vector(512)
2. Re-embeds all closet items using FashionCLIP image encoder
3. Re-embeds all catalog items using FashionCLIP image encoder (with text fallback)
4. Clears taste vectors (they're incompatible across dimensions)
5. Rebuilds HNSW indexes

Run: python scripts/migrate_to_fashionclip.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import httpx
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

from services.fashion_clip import embed_image, embed_text, EMBEDDING_DIM
from services.embedding import build_embedding_text


def migrate():
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL not set")
        return

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cursor = conn.cursor()

    print(f"Migrating to FashionCLIP 2.0 ({EMBEDDING_DIM}-dim embeddings)")
    print("=" * 60)

    # Step 1: Alter vector columns
    print("\n[1/5] Altering vector columns from 1536 → 512...")
    alter_statements = [
        "ALTER TABLE catalog_items ALTER COLUMN embedding TYPE vector(512) USING NULL",
        "ALTER TABLE outfit_generations ALTER COLUMN base_item_embedding TYPE vector(512) USING NULL",
        "ALTER TABLE taste_vectors ALTER COLUMN taste_embedding TYPE vector(512) USING NULL",
        "ALTER TABLE taste_vectors ALTER COLUMN dislike_embedding TYPE vector(512) USING NULL",
        "ALTER TABLE user_closet_items ALTER COLUMN embedding TYPE vector(512) USING NULL",
    ]

    for stmt in alter_statements:
        table = stmt.split("TABLE ")[1].split(" ")[0]
        try:
            cursor.execute(stmt)
            print(f"  {table}: OK")
        except Exception as e:
            print(f"  {table}: {e}")
            conn.rollback()

    conn.commit()

    # Step 2: Clear taste vectors (incompatible dimensions)
    print("\n[2/5] Clearing taste vectors (dimension mismatch)...")
    cursor.execute("UPDATE taste_vectors SET taste_embedding = NULL, dislike_embedding = NULL")
    conn.commit()
    print(f"  Cleared")

    # Step 3: Rebuild indexes
    print("\n[3/5] Rebuilding vector indexes (HNSW)...")
    index_statements = [
        "DROP INDEX IF EXISTS catalog_items_embedding_idx",
        "DROP INDEX IF EXISTS idx_closet_embedding",
        "CREATE INDEX catalog_items_embedding_idx ON catalog_items USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
        "CREATE INDEX idx_closet_embedding ON user_closet_items USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
    ]
    for stmt in index_statements:
        try:
            cursor.execute(stmt)
            print(f"  {stmt[:60]}... OK")
        except Exception as e:
            print(f"  WARN: {e}")
            conn.rollback()
    conn.commit()

    # Step 4: Re-embed closet items
    print("\n[4/5] Re-embedding closet items with FashionCLIP...")
    cursor.execute("SELECT id, name, image_url, category, primary_color, fit, material, style_tags, occasion_tags, season_tags FROM user_closet_items")
    closet_items = cursor.fetchall()
    cols = ["id", "name", "image_url", "category", "primary_color", "fit", "material", "style_tags", "occasion_tags", "season_tags"]

    success, failed = 0, 0
    for row in closet_items:
        item = dict(zip(cols, row))
        item_id = item["id"]
        try:
            embedding = None
            if item.get("image_url"):
                try:
                    resp = httpx.get(item["image_url"], timeout=15.0, follow_redirects=True)
                    if resp.status_code == 200:
                        embedding = embed_image(resp.content)
                except Exception:
                    pass

            if embedding is None:
                text = build_embedding_text(item)
                embedding = embed_text(text)

            cursor.execute("UPDATE user_closet_items SET embedding = %s WHERE id = %s", (embedding, item_id))
            conn.commit()
            success += 1
            print(f"  [{success}/{len(closet_items)}] {item.get('name', 'unknown')}: OK")
        except Exception as e:
            failed += 1
            print(f"  [{item_id}] FAIL: {e}")
            conn.rollback()

    print(f"  Closet: {success} OK, {failed} failed")

    # Step 5: Re-embed catalog items
    print("\n[5/5] Re-embedding catalog items with FashionCLIP...")
    cursor.execute("SELECT id, name, image_url, category, primary_color, fit, material, style_tags, occasion_tags, season_tags FROM catalog_items")
    catalog_items = cursor.fetchall()

    success, failed = 0, 0
    for row in catalog_items:
        item = dict(zip(cols, row))
        item_id = item["id"]
        try:
            embedding = None
            if item.get("image_url"):
                try:
                    resp = httpx.get(item["image_url"], timeout=15.0, follow_redirects=True)
                    if resp.status_code == 200:
                        embedding = embed_image(resp.content)
                except Exception:
                    pass

            if embedding is None:
                text = build_embedding_text(item)
                embedding = embed_text(text)

            cursor.execute("UPDATE catalog_items SET embedding = %s WHERE id = %s", (embedding, item_id))
            conn.commit()
            success += 1
            if success % 50 == 0:
                print(f"  [{success}/{len(catalog_items)}]...")
        except Exception as e:
            failed += 1
            conn.rollback()

    print(f"  Catalog: {success} OK, {failed} failed")

    # Also handle shopify_catalog_items if table exists
    try:
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'shopify_catalog_items')")
        if cursor.fetchone()[0]:
            print("\n[Bonus] Altering shopify_catalog_items...")
            cursor.execute("ALTER TABLE shopify_catalog_items ALTER COLUMN embedding TYPE vector(512) USING NULL")
            conn.commit()
            print("  shopify_catalog_items: column altered (items need re-embedding via Shopify sync)")
    except Exception as e:
        print(f"  shopify_catalog_items: {e}")
        conn.rollback()

    cursor.close()
    conn.close()
    print("\n" + "=" * 60)
    print("Migration complete! FashionCLIP 2.0 is now active.")
    print("All new uploads will use FashionCLIP image embeddings (512-dim).")


if __name__ == "__main__":
    migrate()

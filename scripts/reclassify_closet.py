"""
Re-classify all closet items through single-call vision pipeline + FashionCLIP embedding.
Run: python scripts/reclassify_closet.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import httpx
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

from services.vision import analyze_image
from services.embedding import embed_item_image


def get_original_url(url: str) -> str:
    """Strip transformations to get original image URL."""
    import re
    cleaned = re.sub(
        r'(/upload/).*?(v\d+/)',
        r'\1\2',
        url
    )
    return cleaned


def reclassify_all():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, category, image_url FROM user_closet_items ORDER BY id")
    items = cursor.fetchall()

    print(f"Found {len(items)} items to reclassify\n")

    for item_id, old_name, old_category, image_url in items:
        print(f"\n[{item_id}] {old_name} ({old_category})")

        try:
            original_url = get_original_url(image_url)
            print(f"  Fetching: {original_url[:80]}...")

            response = httpx.get(original_url, timeout=30.0, follow_redirects=True)
            if response.status_code != 200:
                print(f"  FAIL: download {response.status_code}")
                continue

            contents = response.content

            print("  Analyzing image...")
            parsed = analyze_image(contents)
            new_category = parsed.get("category", old_category)
            new_color = parsed.get("primary_color", "unknown")
            new_name = f"{new_color.title()} {new_category.title()}"

            print(f"  New: {new_name} ({new_category})")

            print("  Generating FashionCLIP embedding...")
            embedding = embed_item_image(contents)

            new_url = original_url.replace(
                "/upload/",
                "/upload/e_background_removal/e_trim/c_pad,ar_3:4,b_white/"
            )

            cursor.execute(
                """UPDATE user_closet_items
                   SET name = %s, category = %s, image_url = %s,
                       primary_color = %s, secondary_colors = %s,
                       style_tags = %s, season_tags = %s, occasion_tags = %s,
                       material = %s, fit = %s, embedding = %s
                   WHERE id = %s""",
                (
                    new_name,
                    new_category,
                    new_url,
                    parsed.get("primary_color"),
                    parsed.get("secondary_colors"),
                    parsed.get("style_tags"),
                    parsed.get("season_tags"),
                    parsed.get("occasion_tags"),
                    parsed.get("material"),
                    parsed.get("fit"),
                    embedding,
                    item_id
                )
            )
            conn.commit()

            if new_category != old_category:
                print(f"  OK: {old_category} -> {new_category}")
            else:
                print(f"  OK (category unchanged)")

        except Exception as e:
            print(f"  ERROR: {e}")
            conn.rollback()

    cursor.close()
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    reclassify_all()

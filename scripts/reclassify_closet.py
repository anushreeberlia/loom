"""
Re-classify all closet items through vision/parser pipeline.
Also re-applies image transformations.
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Import the services
from services.vision import describe_image
from services.parser import parse_description
from services.embedding import embed_base_item


def get_original_url(url: str) -> str:
    """Strip transformations to get original image URL."""
    import re
    # Remove everything between /upload/ and /vXXX/ (handles chained transforms with /)
    cleaned = re.sub(
        r'(/upload/).*?(v\d+/)',
        r'\1\2',
        url
    )
    return cleaned


def reclassify_all():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Get all closet items
    cursor.execute("SELECT id, name, category, image_url FROM user_closet_items ORDER BY id")
    items = cursor.fetchall()
    
    print(f"Found {len(items)} items to reclassify\n")
    
    for item_id, old_name, old_category, image_url in items:
        print(f"\n[{item_id}] {old_name} ({old_category})")
        
        try:
            # Get original image URL (without transformations)
            original_url = get_original_url(image_url)
            print(f"  Fetching: {original_url[:80]}...")
            
            # Download image
            response = httpx.get(original_url, timeout=30.0, follow_redirects=True)
            if response.status_code != 200:
                print(f"  ❌ Failed to download: {response.status_code}")
                continue
            
            contents = response.content
            
            # Re-run vision
            print("  Running vision...")
            description = describe_image(contents)
            print(f"  Description: {description[:60]}...")
            
            # Re-run parser
            print("  Parsing...")
            parsed = parse_description(description)
            new_category = parsed.get("category", old_category)
            new_color = parsed.get("primary_color", "unknown")
            new_name = f"{new_color.title()} {new_category.title()}"
            
            print(f"  New: {new_name} ({new_category})")
            
            # Re-generate embedding
            print("  Generating embedding...")
            embedding = embed_base_item(parsed)
            
            # Build new URL with transformations (chained for proper order)
            new_url = original_url.replace(
                "/upload/",
                "/upload/e_background_removal/e_trim/c_pad,ar_3:4,b_white/"
            )
            
            # Update database
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
                print(f"  ✅ Reclassified: {old_category} → {new_category}")
            else:
                print(f"  ✅ Updated (category unchanged)")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            conn.rollback()
    
    cursor.close()
    conn.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    reclassify_all()


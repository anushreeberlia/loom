"""
Reprocess existing closet images with AI enhancements via URL transformation.
Run: python scripts/reprocess_closet.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import re
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def add_transformations_to_url(url: str) -> str:
    """
    Add Cloudinary transformations to existing URL.
    e.g., /upload/v123/closet/img.jpg -> /upload/a_auto,e_background_removal,e_improve,q_auto:best/v123/closet/img.jpg
    """
    # Match Cloudinary URL pattern
    pattern = r'(https://res\.cloudinary\.com/[^/]+/image/upload/)(v\d+/.+)'
    match = re.match(pattern, url)
    
    if match:
        base = match.group(1)
        path = match.group(2)
        transformations = "a_auto,e_background_removal,e_improve,q_auto:best,f_auto"
        return f"{base}{transformations}/{path}"
    
    return url  # Return unchanged if not matching pattern


def reprocess_all():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Get all closet items
    cursor.execute("SELECT id, name, image_url FROM user_closet_items")
    items = cursor.fetchall()
    
    print(f"Found {len(items)} items to reprocess\n")
    
    updated = 0
    for item_id, name, old_url in items:
        # Check if already has auto-rotate
        if "a_auto" in old_url:
            print(f"[{item_id}] {name} - Already has auto-rotate, skipping")
            continue
        
        # If has old transformations without a_auto, add it
        if "e_background_removal" in old_url:
            # Insert a_auto at the beginning of transformations
            new_url = old_url.replace("e_background_removal", "a_auto,e_background_removal")
        else:
            new_url = add_transformations_to_url(old_url)
        
        if new_url != old_url:
            print(f"[{item_id}] {name}")
            print(f"  Old: {old_url}")
            print(f"  New: {new_url}")
            
            cursor.execute(
                "UPDATE user_closet_items SET image_url = %s WHERE id = %s",
                (new_url, item_id)
            )
            conn.commit()
            print(f"  ✅ Updated!")
            updated += 1
        else:
            print(f"[{item_id}] {name} - URL format not recognized, skipping")
    
    cursor.close()
    conn.close()
    print(f"\n✅ Done! Updated {updated} items")


if __name__ == "__main__":
    reprocess_all()

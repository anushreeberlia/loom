#!/usr/bin/env python3
"""Re-tag all closet items with updated parser tags."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Starting retag script...", flush=True)

import httpx
import psycopg2
from dotenv import load_dotenv

print("Loading .env...", flush=True)
load_dotenv()

from services.vision import describe_image
from services.parser import parse_description

DATABASE_URL = os.getenv("DATABASE_URL")
print(f"DATABASE_URL: {DATABASE_URL[:30]}..." if DATABASE_URL else "DATABASE_URL: None", flush=True)

def main():
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL not set in .env", flush=True)
        return
    
    print("Connecting to database...", flush=True)
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    print("Connected!", flush=True)
    
    # Get all items
    cursor.execute("SELECT id, image_url, name FROM user_closet_items")
    rows = cursor.fetchall()
    
    print(f"Re-tagging {len(rows)} items...\n", flush=True)
    
    success = 0
    failed = 0
    
    for row in rows:
        item_id, image_url, old_name = row
        try:
            print(f"[{item_id}] {old_name}...", flush=True)
            
            # Download image
            print(f"  Downloading image...", flush=True)
            response = httpx.get(image_url, timeout=15)
            if response.status_code != 200:
                print(f"  FAILED: Could not download image (status {response.status_code})", flush=True)
                failed += 1
                continue
            
            # Re-run vision + parser
            print(f"  Running vision...", flush=True)
            description = describe_image(response.content)
            print(f"  Vision: {description[:100]}...", flush=True)
            
            print(f"  Parsing...", flush=True)
            parsed = parse_description(description)
            
            # Generate new name
            name = f"{parsed.get('primary_color', '')} {parsed.get('category', 'item')}".strip().title()
            
            # Update database
            print(f"  Updating DB...", flush=True)
            cursor.execute(
                """UPDATE user_closet_items 
                   SET name = %s, category = %s, primary_color = %s, secondary_colors = %s,
                       style_tags = %s, season_tags = %s, occasion_tags = %s, material = %s, fit = %s
                   WHERE id = %s""",
                (
                    name,
                    parsed.get("category", "top"),
                    parsed.get("primary_color"),
                    parsed.get("secondary_colors"),
                    parsed.get("style_tags"),
                    parsed.get("season_tags"),
                    parsed.get("occasion_tags"),
                    parsed.get("material"),
                    parsed.get("fit"),
                    item_id
                )
            )
            conn.commit()
            
            print(f"  ✓ {name}", flush=True)
            print(f"    style: {parsed.get('style_tags')}", flush=True)
            print(f"    occasion: {parsed.get('occasion_tags')}", flush=True)
            success += 1
            
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            failed += 1
    
    cursor.close()
    conn.close()
    
    print(f"\n✓ Done! {success} success, {failed} failed", flush=True)

if __name__ == "__main__":
    main()


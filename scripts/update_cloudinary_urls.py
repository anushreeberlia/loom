"""
Update database image URLs to point to Cloudinary.

Run this AFTER uploading images to Cloudinary via the web UI.

Usage:
    python scripts/update_cloudinary_urls.py
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Your Cloudinary cloud name
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dfvwi4mqd")
FOLDER = "loom"  # The folder you created in Cloudinary

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")


def main():
    print("=" * 50)
    print("Updating Database URLs to Cloudinary")
    print("=" * 50)
    print(f"Cloud: {CLOUD_NAME}")
    print(f"Folder: {FOLDER}")
    print()
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Get all H&M items with local image paths
    cursor.execute("""
        SELECT id, image_url 
        FROM catalog_items 
        WHERE source = 'h_and_m' 
        AND image_url LIKE 'catalog/images/%'
    """)
    items = cursor.fetchall()
    
    print(f"Found {len(items)} items to update\n")
    
    if len(items) == 0:
        print("No items to update. Already migrated?")
        return
    
    updated = 0
    for item_id, local_path in items:
        # Extract filename from local path: catalog/images/0123456789.jpg -> 0123456789.jpg
        filename = local_path.replace("catalog/images/", "")
        
        # Remove .jpg extension for Cloudinary public_id
        public_id = filename.replace(".jpg", "")
        
        # Build Cloudinary URL
        cloudinary_url = f"https://res.cloudinary.com/{CLOUD_NAME}/image/upload/{FOLDER}/{public_id}"
        
        cursor.execute(
            "UPDATE catalog_items SET image_url = %s WHERE id = %s",
            (cloudinary_url, item_id)
        )
        updated += 1
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ Updated {updated} items!")
    print()
    print("Example URL:")
    print(f"  https://res.cloudinary.com/{CLOUD_NAME}/image/upload/{FOLDER}/0123456789")


if __name__ == "__main__":
    main()


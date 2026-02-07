"""
Fix Cloudinary URLs in database by matching actual public IDs to article IDs.
"""
import os
import re
import psycopg2
import cloudinary
import cloudinary.api
from dotenv import load_dotenv

load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")


def get_all_cloudinary_images():
    """Fetch all images from Cloudinary."""
    print("Fetching images from Cloudinary...")
    
    all_resources = []
    next_cursor = None
    
    while True:
        result = cloudinary.api.resources(
            type="upload",
            max_results=500,
            next_cursor=next_cursor
        )
        
        all_resources.extend(result.get("resources", []))
        next_cursor = result.get("next_cursor")
        
        print(f"  Fetched {len(all_resources)} images so far...")
        
        if not next_cursor:
            break
    
    print(f"Total images found: {len(all_resources)}")
    return all_resources


def extract_article_id(public_id):
    """Extract the article ID from a public_id like '0928928001_wtxeyf'."""
    # Remove folder prefix if present
    name = public_id.split("/")[-1]
    # Extract the numeric part before underscore
    match = re.match(r"(\d+)", name)
    if match:
        return match.group(1)
    return None


def build_cloudinary_url(public_id):
    """Build the full Cloudinary URL for a public_id."""
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    return f"https://res.cloudinary.com/{cloud_name}/image/upload/{public_id}"


def update_database_urls(url_mapping):
    """Update database with correct Cloudinary URLs."""
    print(f"\nUpdating {len(url_mapping)} items in database...")
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    updated = 0
    for article_id, cloudinary_url in url_mapping.items():
        # Find items where source_item_id matches the article_id
        cur.execute("""
            UPDATE catalog_items 
            SET image_url = %s 
            WHERE source_item_id = %s
            RETURNING id
        """, (cloudinary_url, article_id))
        
        rows = cur.fetchall()
        if rows:
            updated += len(rows)
            print(f"  Updated {len(rows)} item(s) for article {article_id}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\nTotal items updated: {updated}")
    return updated


def main():
    # Step 1: Get all Cloudinary images
    resources = get_all_cloudinary_images()
    
    # Step 2: Build mapping from article_id -> cloudinary_url
    url_mapping = {}
    for resource in resources:
        public_id = resource.get("public_id", "")
        article_id = extract_article_id(public_id)
        
        if article_id:
            url = build_cloudinary_url(public_id)
            url_mapping[article_id] = url
            
    print(f"\nMapped {len(url_mapping)} article IDs to Cloudinary URLs")
    
    # Show a few examples
    print("\nSample mappings:")
    for i, (article_id, url) in enumerate(list(url_mapping.items())[:3]):
        print(f"  {article_id} -> {url}")
    
    # Step 3: Update database
    if url_mapping:
        update_database_urls(url_mapping)
    else:
        print("No mappings found!")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Post-process closet images: remove backgrounds and re-upload to Cloudinary.
Run locally (not on Railway) to avoid OOM issues.
"""
import os
import io
import psycopg2
import httpx
import cloudinary
import cloudinary.uploader
from PIL import Image
from rembg import remove
from dotenv import load_dotenv

load_dotenv()

# Config
DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dfvwi4mqd"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def process_image(image_url):
    """Download image, remove background, add white bg, return processed bytes."""
    # Download
    response = httpx.get(image_url, timeout=30.0)
    response.raise_for_status()
    
    # Remove background
    input_bytes = response.content
    output_bytes = remove(input_bytes)
    
    # Open and add white background
    img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    
    # Create white background
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
    
    # Resize if too large
    max_dim = 1200
    if max(background.size) > max_dim:
        background.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    
    # Save to bytes
    output_buffer = io.BytesIO()
    background.save(output_buffer, format="JPEG", quality=90)
    output_buffer.seek(0)
    
    return output_buffer.getvalue()

def main():
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    
    # Get all closet items
    cursor.execute("""
        SELECT id, name, category, image_url 
        FROM user_closet_items 
        ORDER BY id
    """)
    items = cursor.fetchall()
    print(f"Found {len(items)} items to process\n")
    
    processed = 0
    errors = 0
    
    for item_id, name, category, image_url in items:
        # Skip if already processed (no transformation params in URL = already clean)
        if "/e_background_removal/" not in image_url and "upload/v" in image_url:
            # Check if it looks like a clean URL already
            print(f"  {item_id}: {name} - might be clean already, processing anyway")
        
        try:
            print(f"{item_id}: {name} ({category})...", end=" ", flush=True)
            
            # Process image
            processed_bytes = process_image(image_url)
            
            # Upload to Cloudinary (just storage, no transformations)
            result = cloudinary.uploader.upload(
                processed_bytes,
                folder="closet_clean",
                resource_type="image"
            )
            new_url = result["secure_url"]
            
            # Update database
            cursor.execute(
                "UPDATE user_closet_items SET image_url = %s WHERE id = %s",
                (new_url, item_id)
            )
            conn.commit()
            
            print(f"✓")
            processed += 1
            
        except Exception as e:
            print(f"ERROR: {e}")
            conn.rollback()
            errors += 1
    
    cursor.close()
    conn.close()
    
    print(f"\nDone! Processed: {processed}, Errors: {errors}")

if __name__ == "__main__":
    main()


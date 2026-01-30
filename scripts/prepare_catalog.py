import csv
import shutil
from pathlib import Path
from collections import defaultdict
import random

# Paths - adjust these to match where you unzipped
INPUT_CSV = Path.home() / "Downloads/archive/styles.csv"
INPUT_IMAGES = Path.home() / "Downloads/archive/images"
OUTPUT_CSV = Path("catalog/items.csv")
OUTPUT_IMAGES = Path("catalog/images")

# Category mapping: Kaggle subCategory → your category
CATEGORY_MAP = {
    "Topwear": "top",
    "Bottomwear": "bottom",
    "Dress": "dress",
    "Shoes": "shoes",
    "Flip Flops": "shoes",
    "Sandal": "shoes",
    "Heels": "shoes",
    "Bags": "accessory",
    "Wallets": "accessory",
    "Clutches": "accessory",
    "Watches": "accessory",
    "Belts": "accessory",
    "Jewellery": "accessory",
    "Sunglasses": "accessory",
    "Scarves": "accessory",
    "Jackets": "layer",
    "Sweaters": "layer",
}

# How many items per category (~700 total)
ITEMS_PER_CATEGORY = {
    "top": 230,
    "bottom": 170,
    "shoes": 140,
    "layer": 80,
    "dress": 50,
    "accessory": 30,
}


def main():
    # Group items by mapped category
    by_category = defaultdict(list)

    with open(INPUT_CSV, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter: Women only
            if row.get("gender", "") != "Women":
                continue
            
            sub_cat = row.get("subCategory", "")
            mapped = CATEGORY_MAP.get(sub_cat)
            
            if not mapped:
                continue  # Skip unmapped categories
            
            item_id = row.get("id", "")
            image_file = INPUT_IMAGES / f"{item_id}.jpg"
            
            if not image_file.exists():
                continue  # Skip if image missing
            
            by_category[mapped].append({
                "id": item_id,
                "name": row.get("productDisplayName", ""),
                "category": mapped,
                "colors": row.get("baseColour", ""),
                "season": row.get("season", ""),
                "occasion": row.get("usage", ""),
            })

    print("Available items per category:")
    for cat, items in by_category.items():
        print(f"  {cat}: {len(items)}")

    # Sample items per category
    selected = []
    for category, limit in ITEMS_PER_CATEGORY.items():
        items = by_category.get(category, [])
        if len(items) > limit:
            items = random.sample(items, limit)
        selected.extend(items)
        print(f"Selected {category}: {len(items)} items")

    # Create output directories
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)

    # Write CSV and copy images
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "category", "image_url", "product_url", "colors", "season", "occasion"])
        
        for item in selected:
            # Copy image
            src = INPUT_IMAGES / f"{item['id']}.jpg"
            dst = OUTPUT_IMAGES / f"{item['id']}.jpg"
            shutil.copy(src, dst)
            
            # Write row
            writer.writerow([
                item["name"],
                item["category"],
                f"catalog/images/{item['id']}.jpg",
                "",  # product_url empty for now
                item["colors"],
                item["season"],
                item["occasion"],
            ])

    print(f"\nDone! Total: {len(selected)} items")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Images: {OUTPUT_IMAGES}")


if __name__ == "__main__":
    main()


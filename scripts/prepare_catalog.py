import csv
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random

# Output paths (same for all datasets)
OUTPUT_CSV = Path("catalog/items.csv")
OUTPUT_IMAGES = Path("catalog/images")

# Dataset configurations
DATASETS = {
    "kaggle": {
        "input_csv": Path.home() / "Downloads/archive/styles.csv",
        "input_images": Path.home() / "Downloads/archive/images",
        "image_pattern": "{id}.jpg",  # How to find source image
        "filter_field": "gender",
        "filter_value": "Women",
        "id_field": "id",
        "name_field": "productDisplayName",
        "category_field": "subCategory",
        "color_field": "baseColour",
        "season_field": "season",
        "occasion_field": "usage",
        "source": "kaggle_fashion",
        "brand": "",
        "category_map": {
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
        },
    },
    "hm": {
        "input_csv": Path.home() / "Downloads/articles.csv",
        "input_images": Path.home() / "Downloads/images",
        "image_pattern": "{id_prefix}/{id}.jpg",  # H&M uses subfolders
        "filter_field": "index_group_name",
        "filter_value": "Ladieswear",
        "id_field": "article_id",
        "name_field": "prod_name",
        "category_field": "product_group_name",
        "color_field": "colour_group_name",
        "season_field": "",  # H&M doesn't have this
        "occasion_field": "",  # H&M doesn't have this
        "source": "h_and_m",
        "brand": "H&M",
        "category_map": {
            "Garment Upper body": "top",
            "Garment Lower body": "bottom",
            "Garment Full body": "dress",
            "Shoes": "shoes",
            "Accessories": "accessory",
            "Bags": "accessory",
            "Items": "accessory",  # misc items
        },
    },
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


def get_image_path(config: dict, item_id: str) -> Path:
    """Build source image path based on dataset pattern."""
    pattern = config["image_pattern"]
    # H&M uses first 3 chars as subfolder
    id_prefix = item_id[:3] if "{id_prefix}" in pattern else ""
    filename = pattern.format(id=item_id, id_prefix=id_prefix)
    return config["input_images"] / filename


def main(dataset_name: str):
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return

    config = DATASETS[dataset_name]
    print(f"Preparing catalog from: {dataset_name}")
    print(f"Input CSV: {config['input_csv']}")

    # Group items by mapped category
    by_category = defaultdict(list)

    with open(config["input_csv"], "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Apply filter (e.g., Women only, Ladieswear only)
            filter_field = config["filter_field"]
            filter_value = config["filter_value"]
            if filter_field and row.get(filter_field, "") != filter_value:
                continue

            # Map category
            raw_category = row.get(config["category_field"], "")
            mapped = config["category_map"].get(raw_category)
            if not mapped:
                continue  # Skip unmapped categories

            # Get item ID and check image exists
            item_id = row.get(config["id_field"], "")
            image_file = get_image_path(config, item_id)
            if not image_file.exists():
                continue  # Skip if image missing

            # Extract fields (with fallbacks for missing columns)
            name = row.get(config["name_field"], "") or f"{raw_category} {item_id}"
            color = row.get(config["color_field"], "") if config["color_field"] else ""
            season = row.get(config["season_field"], "") if config["season_field"] else ""
            occasion = row.get(config["occasion_field"], "") if config["occasion_field"] else ""

            by_category[mapped].append({
                "id": item_id,
                "name": name,
                "category": mapped,
                "colors": color,
                "season": season,
                "occasion": occasion,
                "source": config["source"],
                "brand": config["brand"],
                "image_file": image_file,
            })

    print("\nAvailable items per category:")
    for cat, items in sorted(by_category.items()):
        print(f"  {cat}: {len(items)}")

    # Sample items per category
    selected = []
    for category, limit in ITEMS_PER_CATEGORY.items():
        items = by_category.get(category, [])
        if len(items) > limit:
            items = random.sample(items, limit)
        selected.extend(items)
        print(f"Selected {category}: {len(items)} items")

    if not selected:
        print("\nNo items selected! Check your input paths and filters.")
        return

    # Create output directories
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)

    # Write CSV and copy images
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "category", "image_url", "product_url", "colors", "season", "occasion", "source", "source_item_id", "brand"])

        for item in selected:
            # Copy image
            src = item["image_file"]
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
                item["source"],
                item["id"],
                item["brand"],
            ])

    print(f"\nDone! Total: {len(selected)} items")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Images: {OUTPUT_IMAGES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare catalog from dataset")
    parser.add_argument("--dataset", default="kaggle", choices=DATASETS.keys(),
                        help="Dataset to process (default: kaggle)")
    args = parser.parse_args()
    main(args.dataset)

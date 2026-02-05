import csv
import argparse
import psycopg2

DATABASE_URL = "postgresql://localhost:5432/outfit_styler"
DEFAULT_CSV = "catalog/items.csv"

ALLOWED_CATEGORIES = {"top", "bottom", "dress", "layer", "shoes", "accessory", "bag"}


def main(input_csv: str = DEFAULT_CSV):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print(f"Importing from {input_csv}...")
    
    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)

        inserted = 0
        skipped = 0

        for row in reader:
            name = row.get("name", "").strip()
            category = row.get("category", "").strip().lower()
            image_url = row.get("image_url", "").strip()
            product_url = row.get("product_url", "").strip() or None
            colors = row.get("colors", "").strip() or None
            season = row.get("season", "").strip() or None
            occasion = row.get("occasion", "").strip() or None
            source = row.get("source", "").strip() or None
            source_item_id = row.get("source_item_id", "").strip() or None
            brand = row.get("brand", "").strip() or None
            
            # Parse style_tags and occasion_tags (pipe-separated in CSV)
            style_tags_str = row.get("style_tags", "").strip()
            occasion_tags_str = row.get("occasion_tags", "").strip()
            style_tags = style_tags_str.split("|") if style_tags_str else []
            occasion_tags_csv = occasion_tags_str.split("|") if occasion_tags_str else []

            # Validate: name exists
            if not name:
                print(f"Skipping: missing name")
                skipped += 1
                continue

            # Validate: category is allowed
            if category not in ALLOWED_CATEGORIES:
                print(f"Skipping: invalid category '{category}' for '{name}'")
                skipped += 1
                continue

            # Validate: image_url exists
            if not image_url:
                print(f"Skipping: missing image_url for '{name}'")
                skipped += 1
                continue

            # Check for duplicate (name + image_url)
            cursor.execute(
                "SELECT id FROM catalog_items WHERE name = %s AND image_url = %s",
                (name, image_url)
            )
            if cursor.fetchone():
                print(f"Skipping duplicate: '{name}'")
                skipped += 1
                continue

            # Merge occasion from CSV column and parsed occasion_tags
            final_occasion_tags = occasion_tags_csv.copy()
            if occasion and occasion not in final_occasion_tags:
                final_occasion_tags.append(occasion)
            
            # Insert
            cursor.execute(
                """
                INSERT INTO catalog_items (name, category, image_url, product_url, primary_color, 
                    style_tags, season_tags, occasion_tags, source, source_item_id, brand, tagged_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    name,
                    category,
                    image_url,
                    product_url,
                    colors or None,  # primary_color as text
                    style_tags if style_tags else [],  # style_tags as array
                    [season] if season else [],  # season_tags as array
                    final_occasion_tags if final_occasion_tags else [],  # occasion_tags as array
                    source,
                    source_item_id,
                    brand,
                )
            )
            inserted += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\nDone! Inserted: {inserted}, Skipped: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import catalog CSV to database")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to catalog CSV file")
    args = parser.parse_args()
    main(args.csv)


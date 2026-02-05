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

            # Insert
            cursor.execute(
                """
                INSERT INTO catalog_items (name, category, image_url, product_url, primary_color, season_tags, occasion_tags, source, source_item_id, brand)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    name,
                    category,
                    image_url,
                    product_url,
                    colors or None,  # primary_color as text
                    [season] if season else [],  # season_tags as array
                    [occasion] if occasion else [],  # occasion_tags as array
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


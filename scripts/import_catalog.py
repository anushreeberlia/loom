import csv
import psycopg2

DATABASE_URL = "postgresql://localhost:5432/outfit_styler"
INPUT_CSV = "catalog/items.csv"

ALLOWED_CATEGORIES = {"top", "bottom", "dress", "layer", "shoes", "accessory", "bag"}


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    with open(INPUT_CSV, "r") as f:
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
                INSERT INTO catalog_items (name, category, image_url, product_url, colors, season_tags, occasion_tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    name,
                    category,
                    image_url,
                    product_url,
                    [colors] if colors else [],  # colors as array
                    [season] if season else [],  # season_tags as array
                    [occasion] if occasion else [],  # occasion_tags as array
                )
            )
            inserted += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\nDone! Inserted: {inserted}, Skipped: {skipped}")


if __name__ == "__main__":
    main()


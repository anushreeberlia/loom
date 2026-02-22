"""
Seeds the Shopify dev store with fashion products from the H&M catalog in Railway.

Usage:
    cd /Users/anushreeberlia/loom
    source venv/bin/activate
    python scripts/seed_shopify_dev_store.py
"""

import os
import sys
import time
import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SHOP_DOMAIN = "loom-10146.myshopify.com"
PRODUCTS_TO_CREATE = 30


def get_access_token():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT access_token FROM shopify_stores WHERE shop_domain = %s AND uninstalled_at IS NULL",
            (SHOP_DOMAIN,)
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()
        conn.close()


def get_hm_products(limit: int):
    """Pull diverse fashion items from H&M catalog — one per category."""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    try:
        # Get a mix across categories
        cur.execute(
            """
            SELECT name, category, image_url, product_url, primary_color,
                   style_tags, occasion_tags, material
            FROM catalog_items
            WHERE source = 'h_and_m'
              AND embedding IS NOT NULL
              AND image_url IS NOT NULL
              AND image_url LIKE 'http%%'
              AND category IN ('top', 'bottom', 'dress', 'shoes', 'layer', 'accessory')
            ORDER BY category, RANDOM()
            LIMIT %s
            """,
            (limit,)
        )
        rows = cur.fetchall()
        cols = ["name", "category", "image_url", "product_url", "primary_color",
                "style_tags", "occasion_tags", "material"]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        cur.close()
        conn.close()


def create_shopify_product(access_token: str, item: dict):
    url = f"https://{SHOP_DOMAIN}/admin/api/2025-01/products.json"
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json",
    }

    tags = []
    if item.get("style_tags"):
        tags.extend(item["style_tags"])
    if item.get("occasion_tags"):
        tags.extend(item["occasion_tags"])
    if item.get("primary_color"):
        tags.append(item["primary_color"])

    price_map = {
        "top": "39.99", "bottom": "49.99", "dress": "69.99",
        "shoes": "59.99", "layer": "89.99", "accessory": "29.99",
    }

    payload = {
        "product": {
            "title": item.get("name", "Fashion Item"),
            "product_type": item.get("category", "top").capitalize(),
            "vendor": "H&M",
            "tags": ", ".join(tags),
            "status": "active",
            "variants": [{"price": price_map.get(item.get("category", "top"), "49.99")}],
            "images": [{"src": item["image_url"]}],
        }
    }

    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=20.0)
        if r.status_code == 201:
            return r.json().get("product", {}).get("title")
        else:
            print(f"  Failed ({r.status_code}): {r.text[:120]}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print(f"Fetching access token for {SHOP_DOMAIN}...")
    token = get_access_token()
    if not token:
        print("ERROR: No access token. Make sure the Loom app is installed on the dev store.")
        sys.exit(1)

    print("Fetching H&M products from Railway DB...")
    items = get_hm_products(PRODUCTS_TO_CREATE)
    if not items:
        print("ERROR: No H&M items found. Make sure the H&M catalog is imported.")
        sys.exit(1)

    print(f"Found {len(items)} items. Creating in Shopify (rate limit: ~2/sec)...")
    created = 0
    for i, item in enumerate(items):
        title = item.get("name", "Unknown")[:50]
        print(f"  [{i+1}/{len(items)}] {item['category']:10} | {title}")
        result = create_shopify_product(token, item)
        if result:
            created += 1
        time.sleep(0.6)  # Shopify rate limit

    print(f"\nDone! Created {created}/{len(items)} products.")
    print("Now go to your Loom app dashboard and click 'Sync Catalog'.")


if __name__ == "__main__":
    main()

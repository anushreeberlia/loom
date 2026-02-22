"""
Shopify catalog ingestion and outfit pre-generation.

Fetches products from a merchant's Shopify store, runs them through
the vision + embedding pipeline, and stores results for fast retrieval.
"""

import httpx
import logging
import psycopg2
from psycopg2.extras import Json
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")

SHOPIFY_PRODUCTS_QUERY = """
query GetProducts($cursor: String) {
  products(first: 50, after: $cursor) {
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        id
        title
        productType
        tags
        onlineStoreUrl
        variants(first: 1) {
          edges {
            node {
              id
              price
            }
          }
        }
        images(first: 1) {
          edges {
            node {
              url
            }
          }
        }
      }
    }
  }
}
"""


def fetch_shopify_products(shop_domain: str, access_token: str) -> list[dict]:
    """
    Fetch all products from a Shopify store via GraphQL Admin API.
    Handles pagination automatically.
    """
    url = f"https://{shop_domain}/admin/api/2025-01/graphql.json"
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json",
    }

    all_products = []
    cursor = None

    while True:
        variables = {"cursor": cursor} if cursor else {}
        response = httpx.post(
            url,
            headers=headers,
            json={"query": SHOPIFY_PRODUCTS_QUERY, "variables": variables},
            timeout=30.0,
        )

        if response.status_code != 200:
            raise Exception(f"Shopify API error {response.status_code}: {response.text}")

        data = response.json()
        if "errors" in data:
            raise Exception(f"Shopify GraphQL errors: {data['errors']}")

        products_data = data["data"]["products"]
        edges = products_data["edges"]

        for edge in edges:
            node = edge["node"]
            images = node.get("images", {}).get("edges", [])
            variants = node.get("variants", {}).get("edges", [])

            image_url = images[0]["node"]["url"] if images else None
            if not image_url:
                continue  # Skip products with no image

            price = None
            variant_id = None
            if variants:
                price = float(variants[0]["node"]["price"])
                variant_id = variants[0]["node"]["id"]

            all_products.append({
                "shopify_product_id": node["id"],
                "shopify_variant_id": variant_id,
                "name": node["title"],
                "product_type": node.get("productType", ""),
                "tags": node.get("tags", []),
                "image_url": image_url,
                "product_url": node.get("onlineStoreUrl"),
                "price": price,
            })

        page_info = products_data["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]

    logger.info(f"Fetched {len(all_products)} products from {shop_domain}")
    return all_products


def upsert_shopify_catalog_item(shop_domain: str, product: dict) -> int:
    """
    Insert or update a Shopify product in shopify_catalog_items.
    Returns the row ID.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO shopify_catalog_items
                (shop_domain, shopify_product_id, shopify_variant_id, name,
                 image_url, product_url, price)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (shop_domain, shopify_product_id)
            DO UPDATE SET
                name = EXCLUDED.name,
                image_url = EXCLUDED.image_url,
                product_url = EXCLUDED.product_url,
                price = EXCLUDED.price,
                shopify_variant_id = EXCLUDED.shopify_variant_id
            RETURNING id
            """,
            (
                shop_domain,
                product["shopify_product_id"],
                product.get("shopify_variant_id"),
                product["name"],
                product["image_url"],
                product.get("product_url"),
                product.get("price"),
            ),
        )
        row_id = cursor.fetchone()[0]
        conn.commit()
        return row_id
    finally:
        cursor.close()
        conn.close()


def save_processed_item(item_id: int, category: str, description: str,
                        base_item: dict, embedding: list[float]):
    """Save vision/parser/embedding results for a catalog item."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            UPDATE shopify_catalog_items SET
                category = %s,
                description = %s,
                primary_color = %s,
                secondary_colors = %s,
                style_tags = %s,
                season_tags = %s,
                occasion_tags = %s,
                material = %s,
                fit = %s,
                embedding = %s,
                processed_at = NOW(),
                processing_error = NULL
            WHERE id = %s
            """,
            (
                category,
                description,
                base_item.get("primary_color"),
                base_item.get("secondary_colors", []),
                base_item.get("style_tags", []),
                base_item.get("season_tags", []),
                base_item.get("occasion_tags", []),
                base_item.get("material"),
                base_item.get("fit"),
                embedding,
                item_id,
            ),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def save_processing_error(item_id: int, error: str):
    """Mark a catalog item as failed processing."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE shopify_catalog_items SET processing_error = %s WHERE id = %s",
            (error, item_id),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def get_all_catalog_item_stubs(shop_domain: str) -> list[dict]:
    """Get all catalog items (id, shopify_product_id, name, image_url) for reprocessing."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, shopify_product_id, name, image_url
            FROM shopify_catalog_items
            WHERE shop_domain = %s
            ORDER BY id
            """,
            (shop_domain,),
        )
        rows = cursor.fetchall()
        return [
            {"id": r[0], "shopify_product_id": r[1], "name": r[2], "image_url": r[3]}
            for r in rows
        ]
    finally:
        cursor.close()
        conn.close()


def get_unprocessed_items(shop_domain: str, limit: int = 50) -> list[dict]:
    """Get catalog items that haven't been processed yet."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, shopify_product_id, name, image_url
            FROM shopify_catalog_items
            WHERE shop_domain = %s
              AND processed_at IS NULL
              AND processing_error IS NULL
            LIMIT %s
            """,
            (shop_domain, limit),
        )
        rows = cursor.fetchall()
        return [
            {"id": r[0], "shopify_product_id": r[1], "name": r[2], "image_url": r[3]}
            for r in rows
        ]
    finally:
        cursor.close()
        conn.close()


def get_shopify_catalog_items(shop_domain: str, category: str = None,
                               limit: int = 200) -> list[dict]:
    """
    Retrieve processed catalog items for outfit retrieval.
    Mirrors the structure expected by the existing retrieval service.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        query = """
            SELECT id, shopify_product_id, name, category, image_url, product_url,
                   price, primary_color, secondary_colors, style_tags, season_tags,
                   occasion_tags, material, fit, embedding::text
            FROM shopify_catalog_items
            WHERE shop_domain = %s
              AND processed_at IS NOT NULL
              AND embedding IS NOT NULL
        """
        params = [shop_domain]
        if category:
            query += " AND category = %s"
            params.append(category)
        query += " LIMIT %s"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            {
                "id": r[0],
                "shopify_product_id": r[1],
                "name": r[2],
                "category": r[3],
                "image_url": r[4],
                "product_url": r[5],
                "price": float(r[6]) if r[6] else None,
                "primary_color": r[7],
                "secondary_colors": r[8] or [],
                "style_tags": r[9] or [],
                "season_tags": r[10] or [],
                "occasion_tags": r[11] or [],
                "material": r[12],
                "fit": r[13],
                "embedding": r[14],
            }
            for r in rows
        ]
    finally:
        cursor.close()
        conn.close()


def save_generated_outfits(shop_domain: str, shopify_product_id: str,
                            outfits: list[dict]):
    """
    Store pre-generated outfits for a product.
    outfits: list of {direction, outfit_items, collage_url, explanation}
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        for outfit in outfits:
            cursor.execute(
                """
                INSERT INTO shopify_generated_outfits
                    (shop_domain, shopify_product_id, direction, outfit_items, collage_url, explanation)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (shop_domain, shopify_product_id, direction)
                DO UPDATE SET
                    outfit_items = EXCLUDED.outfit_items,
                    collage_url = EXCLUDED.collage_url,
                    explanation = EXCLUDED.explanation,
                    generated_at = NOW()
                """,
                (
                    shop_domain,
                    shopify_product_id,
                    outfit["direction"],
                    Json(outfit["outfit_items"]),
                    outfit.get("collage_url"),
                    outfit.get("explanation"),
                ),
            )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def get_generated_outfits(shop_domain: str, shopify_product_id: str) -> list[dict]:
    """Retrieve pre-generated outfits for a product page."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT direction, outfit_items, collage_url, explanation, generated_at
            FROM shopify_generated_outfits
            WHERE shop_domain = %s AND shopify_product_id = %s
            ORDER BY direction
            """,
            (shop_domain, shopify_product_id),
        )
        rows = cursor.fetchall()
        return [
            {
                "direction": r[0],
                "outfit_items": r[1],
                "collage_url": r[2],
                "explanation": r[3],
                "generated_at": r[4].isoformat() if r[4] else None,
            }
            for r in rows
        ]
    finally:
        cursor.close()
        conn.close()


def update_store_stats(shop_domain: str):
    """Update product + outfit counts for a store."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            UPDATE shopify_stores SET
                product_count = (
                    SELECT COUNT(*) FROM shopify_catalog_items
                    WHERE shop_domain = %s AND processed_at IS NOT NULL
                ),
                outfit_count = (
                    SELECT COUNT(DISTINCT shopify_product_id)
                    FROM shopify_generated_outfits
                    WHERE shop_domain = %s
                ),
                catalog_synced_at = NOW()
            WHERE shop_domain = %s
            """,
            (shop_domain, shop_domain, shop_domain),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

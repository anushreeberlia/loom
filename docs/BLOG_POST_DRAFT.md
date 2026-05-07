# Loom: personal closet app + Shopify “Shop the Look”

*Draft — replace `[PARTNER_INSTALL_URL]` with your Shopify Partners install URL before publishing.*

---

## Personal app

Loom is a web app where users **sign in** (Google OAuth or email and password), then **manage a closet**: upload photos of clothing, get automatic tagging (category, color, material, style, season, occasion), and optional background handling on upload.

From that closet the system **generates outfits**: three stylistic directions per request, built from the user’s own items. Users get **daily outfit suggestions** that factor in **weather** and time of day. They can type a **mood or occasion** in plain language (e.g. job interview, beach day); text is embedded and matched semantically to surface relevant combinations.

**Style** is not static: **like/dislike feedback** on suggestions updates a personal taste vector that re-ranks future outfits. There is **rotation** so the same pieces are not repeated too often. Users can **save** outfits and **log what they wore**.

Auth is JWT-backed; closet and profile data live in Postgres with the rest of the backend.

## Shopify app

Merchants install an embedded **Admin app** (`loom-app`, Node/React Router + Prisma sessions). They **sync the store catalog** from Shopify Admin API; each product image runs through the same **vision + embedding** stack as the personal app.

On the storefront, a **theme extension** adds a **“Shop the Look”** block on product pages. It calls the Python API for **precomputed outfits** for that product. The three suggested looks are **only assembled from that merchant’s catalog**—the synced products for that shop. No cross-store data, no external catalog.

Admin traffic (OAuth, webhooks, “sync”) goes to the Node service. Storefront outfit requests go to the **FastAPI** host the merchant configures in the block (public HTTPS). Production typically uses **two deployed URLs**: one for the embedded app origin, one for the API the theme can fetch.

**Install (Partner link):** `[PARTNER_INSTALL_URL]`

After install: sync catalog, paste the API base URL into the block settings, add the block to the product template.

## How outfits are computed (both products)

Vision uses **Fashion Florence**: Florence-2 **fine-tuned** on fashion imagery to produce structured attributes instead of generic captions. **GPT-4o-mini** is used when Florence is unavailable or as a fallback path.

**FashionCLIP** (ONNX, 512-dimensional) embeds images and text. **Postgres + pgvector** stores embeddings; retrieval uses cosine similarity with **slot constraints** (e.g. top, bottom, shoes) so outfits stay structurally valid. The three directions reuse the same candidate pool with **different scoring weights** per direction.

Collages for Shopify caching use **Cloudinary** where applicable.

## Repo

MIT-licensed source: omit or add your repo URL depending on whether you want the post linked to a GitHub account.

Use Shopify’s **partner brand guidelines** if you display Shopify trademarks.

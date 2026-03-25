# Loom Style

AI-powered outfit engine that analyzes clothing images, generates complete outfit suggestions, and learns your style preferences over time.

Two modes: a **personal wardrobe assistant** (upload your closet, get daily outfits) and a **Shopify app** that generates "Shop the Look" outfit suggestions on product pages from a merchant's catalog.

## How It Works

```
Upload image → Fashion Florence (vision) → structured tags (category, color, material, style)
                                         → FashionCLIP embedding (512-dim)
                                         → pgvector similarity search
                                         → outfit assembly (Classic / Trendy / Bold)
                                         → collage generation
```

1. **Vision analysis** — Fashion Florence (fine-tuned Florence-2) extracts category, color, material, and style tags from clothing images. Falls back to GPT-4o-mini if Florence is unavailable. FashionCLIP provides zero-shot color classification when Florence returns unknown.

2. **Embedding** — FashionCLIP 2.0 (ONNX, 512-dim) generates embeddings that capture visual style, not just category labels. Embeddings blend image features with text metadata for richer retrieval.

3. **Retrieval** — pgvector HNSW index finds complementary items by cosine similarity, filtered by category slots (top + bottom + shoes + optional layer/accessory).

4. **Scoring** — Candidates are ranked by color harmony, style coherence, occasion match, season/weather fit, and user taste vectors (learned from like/dislike feedback).

5. **Assembly** — Three style directions (Classic, Trendy, Bold) are generated per input item. Each direction uses different scoring weights to produce distinct outfit aesthetics.

## Features

### Personal App

- **Closet management** — Upload items with automatic AI tagging and background removal
- **Daily outfits** — 3 weather-aware, occasion-appropriate outfit suggestions per day
- **Mood input** — Type any mood ("cozy brunch", "job interview", "date night") and get matching outfits
- **Style learning** — Like/dislike feedback builds a personal taste vector that improves suggestions
- **Top rotation** — FIFO queue prevents showing the same items repeatedly
- **Save and track** — Bookmark outfits and log what you've worn

### Shopify App

- **Catalog sync** — Imports all products via Shopify GraphQL API, processes through the same vision+embedding pipeline
- **Pre-generated outfits** — Every product gets 3 outfit directions cached with Cloudinary collages
- **Theme extension** — "Shop the Look" Liquid block renders outfit cards on any product page
- **Incremental updates** — New products trigger smart invalidation: only regenerates outfits for categories affected by the new item
- **Webhook-driven** — `products/create` automatically processes new items in the background

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.10+ |
| Database | PostgreSQL 15+ with pgvector (HNSW, cosine) |
| Vision | Fashion Florence (fine-tuned Florence-2) via HuggingFace Space |
| Embeddings | FashionCLIP 2.0 (ONNX runtime, 512-dim) |
| Fallback vision | GPT-4o-mini (OpenAI) |
| Images | Cloudinary |
| Weather | OpenWeatherMap API |
| Auth | Google OAuth + email/password with JWT |
| Shopify admin UI | React Router + Shopify App Bridge (`loom-app/`) |
| Hosting | Railway |

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ with pgvector extension
- Cloudinary account
- OpenAI API key (for Shopify app and Florence fallback)

### Setup

```bash
git clone https://github.com/anushreeberlia/loom.git
cd loom

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp env.example .env
# Edit .env with your API keys

createdb loom
psql loom -f schema.sql

uvicorn app:app --reload --port 8080
```

Visit http://localhost:8080

### Using the Personal App

1. **Sign up** — Google OAuth or email/password at `/login`
2. **Upload items** — Go to `/inventory`, photograph or upload your clothing. Background removal runs client-side. The AI tags each item automatically (category, color, material, style, occasion, season).
3. **Get daily outfits** — Visit `/closet` for 3 daily suggestions. They adapt to your local weather and time of day (work hours → work outfits, evenings → going-out).
4. **Try a mood** — Type anything in the mood box: "casual friday", "beach vacation", "all black". The system embeds your text with FashionCLIP and matches it semantically.
5. **Give feedback** — Like or dislike outfits. This trains your personal taste vector, which re-ranks future suggestions.
6. **Generate from an item** — Click any closet item to generate 3 outfits built around it.

### Setting Up the Shopify App

Requires Node.js 20+ and [Shopify CLI](https://shopify.dev/docs/apps/tools/cli).

```bash
cd loom-app
npm install
npm run dev
```

This starts the embedded admin app with a Cloudflare tunnel. The theme extension (`shop-the-look`) deploys to Shopify via `shopify app deploy`.

The Shopify backend routes (`/shopify/*`) are mounted into the main FastAPI app — no separate deployment needed for the API. The Node admin UI (`loom-app/`) needs its own hosting in production (e.g. a second Railway service).

Set `LOOM_BACKEND_URL` in the Node app's environment to point at your Python API.

## Environment Variables

See `env.example` for the full list:

```
DATABASE_URL=postgresql://user:pass@localhost/loom
VISION_BACKEND=florence              # or "openai"
FLORENCE_API_URL=https://your-florence-space.hf.space
OPENAI_API_KEY=sk-...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
JWT_SECRET=change-me
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
OPENWEATHERMAP_API_KEY=...
SHOPIFY_API_SECRET=shpss_...         # only for Shopify webhook verification
```

## API Endpoints

### Personal App

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/closet` | Daily outfits page |
| GET | `/inventory` | Closet management page |
| GET | `/v1/closet/items` | List closet items |
| POST | `/v1/closet/items` | Upload a new item (image + auto-tag) |
| DELETE | `/v1/closet/items/{id}` | Remove an item |
| GET | `/v1/closet/daily` | Get daily outfit recommendations |
| POST | `/v1/closet/outfits:generate` | Generate outfits from a specific item |
| POST | `/v1/closet/feedback` | Submit like/dislike feedback |
| POST | `/v1/closet/outfits/save` | Bookmark an outfit |
| GET | `/v1/closet/outfits/saved` | List saved outfits |
| GET | `/v1/closet/outfits/worn` | Outfit wear history |
| GET | `/v1/weather` | Current weather data |

### Shopify

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/shopify/install` | Store access token after OAuth |
| POST | `/shopify/catalog/sync` | Full catalog fetch + process |
| POST | `/shopify/catalog/resync` | Re-sync using stored token |
| POST | `/shopify/catalog/reprocess` | Re-run vision+embed on all items |
| GET | `/shopify/catalog/status` | Product/outfit counts |
| GET | `/shopify/outfits` | Get outfits for a product page |
| POST | `/shopify/outfits/generate` | Trigger outfit generation |
| POST | `/shopify/webhooks/product_created` | New product webhook |

## Project Structure

```
loom/
├── app.py                    # Main FastAPI app (personal routes + mounts shopify_app)
├── shopify_app.py            # Shopify API backend (catalog, outfits, webhooks)
├── schema.sql                # Full database schema (personal + Shopify tables)
├── requirements.txt          # Python dependencies
├── env.example               # Environment variable template
├── services/
│   ├── fashion_florence.py   # Florence vision API client + FashionCLIP color fallback
│   ├── fashion_clip.py       # FashionCLIP 2.0 ONNX embeddings + zero-shot classify
│   ├── vision.py             # Vision router (Florence vs OpenAI)
│   ├── tagging.py            # Tag validation and normalization
│   ├── item_processor.py     # Unified vision + embedding pipeline
│   ├── outfit_generator.py   # Outfit generation orchestrator
│   ├── retrieval.py          # pgvector similarity search + scoring
│   ├── collage.py            # Outfit image collage generation
│   ├── weather.py            # OpenWeatherMap integration
│   ├── auth.py               # JWT + Google OAuth
│   └── shopify_catalog.py    # Shopify GraphQL product fetch + DB helpers
├── static/
│   ├── closet.html           # Daily outfits UI
│   ├── inventory.html        # Closet management UI
│   ├── index.html            # Demo / single-item generation
│   ├── landing.html          # Landing page
│   └── login.html            # Auth page
├── space/                    # HuggingFace Space for Fashion Florence API
│   ├── app.py                # FastAPI: loads model, /analyze endpoint
│   ├── Dockerfile
│   └── requirements.txt
├── loom-app/                 # Shopify embedded admin app (Node.js)
│   ├── app/routes/           # React Router routes (admin UI, webhooks)
│   ├── extensions/shop-the-look/  # Theme extension (Liquid block)
│   └── shopify.app.toml      # Shopify Partners config
├── training/                 # Florence-2 fine-tuning scripts
└── scripts/                  # Data import and maintenance utilities
```

## License

MIT

# Loom — Outfit Builder & Shopify “Shop the Look”

Your personal AI-powered wardrobe assistant **plus** a **Shopify app** that ingests a merchant’s catalog and shows **AI outfit suggestions** on product pages (“Shop the Look”).

---

## What lives in this repository

| Part | Folder / entrypoint | Who uses it | Role |
|------|---------------------|-------------|------|
| **Consumer web app** | `app.py`, `static/` | You (signed-in user) | Closet, daily outfits, weather, feedback |
| **Shopify API backend** | `shopify_app.py` (mounted from `app.py`) | Merchant’s **storefront theme** + **loom-app** server | Catalog sync, `/shopify/outfits`, webhooks |
| **Shopify Admin app** | `loom-app/` (Node, React Router) | **Merchant staff** in Shopify Admin | OAuth, sessions, “Sync catalog,” status UI |
| **Theme extension** | `loom-app/extensions/shop-the-look/` | **Shoppers** on product pages | Renders outfits; calls your **Python** API URL |

There are **two different “apps”** in the Shopify sense:

1. **Python on Railway** (e.g. `https://loom-style.com`) — the **outfit + catalog engine** the theme talks to.
2. **`loom-app` (Node)** — the **embedded Admin UI** Shopify opens in an iframe; it needs its **own** stable `https://` URL in production (often a **second** Railway service).

---

## Architecture (literal diagrams)

### A) Merchant opens “Loom” in Shopify Admin

Staff work **inside Shopify**. Shopify loads **your Node app** in an iframe. That Node server then calls **your Python API** (server-to-server) for install/sync/status.

```
┌─────────────────────────────────────────────────────────────┐
│  Browser: merchant.shopify.com (Shopify Admin)               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  iframe: your NODE app (loom-app)                      │ │
│  │  e.g. https://app.example.com/app?embedded=1&shop=…     │ │
│  │  • OAuth / session stored in Prisma (DB)                 │ │
│  │  • “Sync catalog” → HTTP POST to Python                   │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         │ iframe URL = SHOPIFY_APP_URL       │ server-side fetch
         ▼                                    ▼
┌──────────────────┐              ┌──────────────────────────┐
│  NODE (loom-app) │ ──────────►  │  PYTHON (e.g. loom-style) │
│  React Router    │  LOOM_       │  FastAPI shopify_app      │
│  Prisma sessions │  BACKEND_URL │  catalog + outfits DB     │
└──────────────────┘              └──────────────────────────┘
```

### B) Shopper views a product page (storefront)

The **theme block** runs on the **store**. It fetches outfits from **Python only** — shoppers **do not** load the Node Admin app.

```
┌─────────────────────────────────────────────────────────────┐
│  Browser: customer on mystore.com/products/some-shirt        │
│  Theme section “Shop the Look”                                │
│       │                                                       │
│       │  JavaScript: fetch(PYTHON_URL + "/shopify/outfits?…")   │
│       ▼                                                       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  PYTHON API              │
│  (theme setting:         │
│   “Loom backend URL”)    │
└──────────────────────────┘
```

### C) Development (typical)

```
Shopify CLI tunnel (*.trycloudflare.com)  ──►  NODE (loom-app)  local/Vite
                                                      │
                                                      │ LOOM_BACKEND_URL
                                                      ▼
                                              PYTHON (Railway or local)
```

### D) Production (typical)

```mermaid
flowchart LR
  subgraph admin [Merchant / Shopify Admin]
    A[Browser iframe]
  end
  subgraph store [Customer / Storefront]
    S[Product page theme]
  end
  subgraph your_infra [Your hosting]
    N[loom-app Node HTTPS]
    P[Python API HTTPS]
    DB1[(Postgres Prisma sessions)]
    DB2[(Postgres Loom catalog)]
  end
  A --> N
  N --> DB1
  N -->|install sync status| P
  S -->|outfits JSON| P
  P --> DB2
```

---

## Repository layout (high level)

```
loom/
├── app.py                 # FastAPI: consumer routes + mounts shopify_app
├── shopify_app.py         # Shopify REST: install, sync, outfits, webhooks
├── services/              # Vision, embeddings, retrieval, shopify_catalog, …
├── env.example            # Python env template
├── loom-app/              # Shopify embedded app + theme extension (see below)
│   ├── app/               # React Router routes, shopify.server.ts
│   ├── extensions/shop-the-look/
│   ├── prisma/            # Session storage (SQLite dev; use Postgres in prod)
│   ├── shopify.app.toml   # Partners config, scopes, webhooks
│   └── .env.example       # Node env template
└── README.md              # This file
```

---

## Shopify: `loom-app` quick start

1. Install [Shopify CLI](https://shopify.dev/docs/apps/tools/cli).
2. From `loom-app/`:

   ```bash
   cd loom-app
   npm install
   npm run dev
   ```

3. Use **Preview (`p`)** in the CLI to open the app in Admin (avoids manual tunnel login when possible).
4. Theme editor: add **Shop the Look** block; set **Loom backend URL** to your **Python** API (e.g. `https://loom-style.com`).

More detail, deploy notes, and Shopify troubleshooting: **`loom-app/README.md`**.

**Important auth/login fix:** the `auth.login` action must **not** forward the incoming `Content-Type` / `Content-Length` headers when rebuilding the `Request` passed to Shopify’s `login()` (see `loom-app/app/routes/auth.login/route.tsx`), or `shop` is dropped from the body.

---

## Features (consumer app)

- **Personal Closet** - Upload and manage your wardrobe items with automatic tagging
- **Daily Outfits** - Get 3 curated outfit suggestions each day
- **Weather-Aware** - Recommendations adapt to local weather conditions
- **Occasion Detection** - Auto-detects work hours, evenings, weekends for appropriate styling
- **Custom Moods** - Type any mood/occasion for personalized suggestions
- **Style Learning** - Like/dislike feedback improves recommendations over time
- **Top Rotation** - FIFO queue ensures variety in outfit suggestions
- **Save & Track** - Bookmark outfits and track what you've worn
- **Background Removal** - Client-side AI removes backgrounds from item photos

## Tech stack

- **Backend**: FastAPI + Python (`app.py`, `shopify_app.py`)
- **Shopify Admin UI**: React Router + `@shopify/shopify-app-react-router` (`loom-app/`)
- **Database**: PostgreSQL + pgvector (consumer + Shopify catalog); Prisma (Shopify **sessions** in `loom-app`, SQLite in dev)
- **AI**: OpenAI / Florence / Fashion CLIP (see `services/`)
- **Images**: Cloudinary
- **Weather**: OpenWeatherMap API
- **Auth (consumer)**: Google OAuth + email/password + JWT
- **Hosting**: Railway (common setup: **Python** service + separate **Node** service for `loom-app` in production)

## Quick Start (Python consumer API)

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ with pgvector extension
- Cloudinary account
- OpenAI API key
- OpenWeatherMap API key (free tier works)

### Setup

```bash
# Clone
git clone https://github.com/anushreeberlia/loom.git
cd loom

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Environment variables
cp env.example .env
# Edit .env with your API keys

# Database
createdb loom
psql loom -f schema.sql

# Run
uvicorn app:app --reload --port 8080
```

Visit http://localhost:8080

## Environment Variables (Python)

See `env.example` for Shopify-related keys (`SHOPIFY_API_SECRET`, optional shared secret for uninstall notify, etc.).

```
DATABASE_URL=postgresql://user:pass@localhost/loom
OPENAI_API_KEY=sk-...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
OPENWEATHERMAP_API_KEY=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
JWT_SECRET=...
```

## API Endpoints (selection)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/closet` | GET | Daily outfits page |
| `/inventory` | GET | Manage closet items |
| `/v1/closet/items` | GET/POST | List/add closet items |
| `/v1/closet/daily` | GET | Get daily outfit recommendations |
| `/v1/closet/outfits:generate` | POST | Generate outfits from a specific item |
| `/v1/closet/feedback` | POST | Submit like/dislike feedback |
| `/v1/closet/outfits/save` | POST | Save outfit for later |
| `/v1/closet/outfits/saved` | GET | List saved outfits |
| `/v1/closet/outfits/worn` | GET | List worn outfit history |
| `/auth/google` | GET | Google OAuth login |
| `/auth/register` | POST | Email/password registration |
| `/auth/login` | POST | Email/password login |
| `/shopify/*` | various | Shopify integration (install, catalog sync, outfits, webhooks) — see `shopify_app.py` |

## Project Structure

```
├── app.py              # Main FastAPI application
├── shopify_app.py      # Shopify FastAPI sub-app
├── schema.sql          # Database schema
├── requirements.txt    # Python dependencies
├── services/
│   ├── retrieval.py    # Outfit retrieval & assembly
│   ├── collage.py      # Outfit image generation
│   ├── weather.py      # Weather API integration
│   ├── auth.py         # Authentication
│   ├── shopify_catalog.py
│   └── ...
├── loom-app/           # Shopify Admin app + theme extension
└── static/
    ├── closet.html     # Daily outfits UI
    ├── inventory.html  # Closet management UI
    ├── index.html      # Demo page
    └── login.html      # Auth page
```

## License

MIT

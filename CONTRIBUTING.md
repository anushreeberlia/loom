# Contributing to Loom

Thanks for your interest. This repo is the **consumer outfit app** (Python/FastAPI), **Shopify catalog + outfits API**, **embedded Admin app** (`loom-app/`), and **theme extension**.

## Before you start

- Copy **`env.example`** → **`.env`** at the repo root for Python.
- In **`loom-app/`**, copy **`.env.example`** → **`.env`** (Shopify CLI also injects `SHOPIFY_*` in dev).
- Never commit real API keys, `SHOPIFY_API_SECRET`, database URLs with passwords, or JWT secrets.

## Local development

**Python API** (from repo root):

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

**Shopify Admin app** (separate terminal):

```bash
cd loom-app
npm install
npm run dev
```

Use Shopify CLI’s preview flow so Admin loads your tunnel URL. Point **`LOOM_BACKEND_URL`** at your local or hosted Python API.

## Pull requests

- Keep changes focused; match existing style.
- Run **`npm run typecheck`** in `loom-app/` when you touch TypeScript.
- If you change Prisma schema, include migrations and note any production baseline steps (`loom-app/prisma/BASELINE.md`).

## Security

Do not open public issues for undisclosed vulnerabilities. Email the maintainer or use GitHub **Security → Report a vulnerability** if enabled.

## Name

This project is **not affiliated** with Loom (Atlassian) or other products named “Loom.”

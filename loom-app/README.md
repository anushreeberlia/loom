# Loom — Shopify app (`loom-app`)

This folder is the **Shopify embedded Admin app** (React Router) plus the **“Shop the Look”** theme app extension for the **Loom** monorepo.

**Read the repository overview and diagrams first:** [../README.md](../README.md) (architecture, two HTTPS endpoints, who calls what).

---

## What this package does

| Surface | Purpose |
|---------|---------|
| **Embedded app** (`/app`, …) | Merchant installs the app, **syncs catalog**, sees status. Server-side calls go to **`LOOM_BACKEND_URL`** (the Python API in the repo root). |
| **Theme extension** | On **product pages**, fetches outfits from the **same Python URL** configured in the block (“Loom backend URL”). |
| **Auth / webhooks** | OAuth and Shopify webhooks hit **this Node app’s** public URL (`SHOPIFY_APP_URL`), not the Python app. |

---

## Local development

```bash
cd loom-app
npm install
npm run dev
```

- Use **Preview (`p`)** in the Shopify CLI TUI to open Admin with the correct store and tunnel.
- Copy **`.env.example`** → `.env` for local overrides (e.g. `LOOM_BACKEND_URL`).

### Auth login (`/auth/login`)

Uses a **native HTML form + `fetcher.submit(FormData)`** and the action **must not** forward incoming `Content-Type` / `Content-Length` when building the `Request` passed to Shopify `login()` — see `app/routes/auth.login/route.tsx`.

### Polaris on standalone pages

`app/root.tsx` loads `polaris.js` + `shopify-api-key` meta so Polaris web components work outside the Admin iframe where needed.

---

## Production

1. Host **this Node app** on its **own** stable `https://` origin (second service if Python is already on Railway).
2. Set **`SHOPIFY_APP_URL`** and Partners **`application_url`** / **`redirect_urls`** to that origin.
3. Provision **PostgreSQL** (e.g. Railway Postgres) and set **`DATABASE_URL`**, **`LOOM_BACKEND_URL`**, **`SCOPES`**, **`SHOPIFY_API_KEY`**, **`SHOPIFY_API_SECRET`**. Run **`npm run setup`** on deploy (migrate). Session storage uses **`public."Session"`** (same Postgres as other services is fine). If the DB already has tables and **`migrate deploy`** fails with **P3005**, follow **`prisma/BASELINE.md`** once. Local DB: `docker compose up -d` then use the URL in **`.env.example`**.
4. Optional: set your host **health check** to **`GET /health`**.
5. **`shopify app deploy`** to push config + extension.

Official Shopify hosting docs: [Deploy a Shopify app](https://shopify.dev/docs/apps/launch/deployment).

---

## Test with a Partner development store

You do **not** need a separate “test account” for Shopify: use a **[development store](https://help.shopify.com/en/partners/dashboard/development-stores)** under your [Shopify Partners](https://partners.shopify.com) account.

1. **Partners** → **Stores** → **Add store** → **Create development store** (fill in store name, purpose).
2. **Partners** → **Apps** → select **loom-app** → **Test on development store** (or use the app install URL from the app’s **Overview**).
3. Open the app in Admin — the **Home** page has a **Getting started** checklist: sync catalog, **copy Loom backend URL**, **open theme editor**, add the **Shop the Look** block, paste the URL in block settings.
4. Your Python API (`LOOM_BACKEND_URL`) must be **public HTTPS** (e.g. Railway) so the **theme block** on the storefront can call `/shopify/outfits`.

Local dev: run **`npm run dev`** in `loom-app/` and use the CLI tunnel URL; set **`LOOM_BACKEND_URL`** to your tunneled or deployed Python API.

---

## Template reference

Scaffolded from [Shopify’s React Router app template](https://github.com/Shopify/shopify-app-template-react-router). Package docs: [@shopify/shopify-app-react-router](https://shopify.dev/docs/api/shopify-app-react-router).

Below is the upstream troubleshooting section (still useful).

---

### Database tables don't exist

Run `npm run setup` (Prisma generate + migrate).

### `migrate deploy` fails with P3005 (non-empty database)

Use **`prisma/BASELINE.md`**: apply both migration SQL files (or `./scripts/railway-baseline.sh`), then `npm run migrate:resolve-baseline` with `DATABASE_URL` set.

### Navigating/redirecting breaks an embedded app

1. Use `Link` from `react-router`, not raw `<a>`.
2. Use redirects from `authenticate.admin`, not ad-hoc `redirect` from react-router for OAuth flows.
3. Embedded session + iframe: follow [Shopify embedded app auth](https://shopify.dev/docs/apps/build/authentication-authorization).

### "nbf" claim timestamp check failed

Sync your computer’s clock (automatic date/time).

### Cloudflare tunnel + streaming

Dev tunnels may buffer streamed responses; production behavior differs.

### Webhooks

Prefer **app-specific** webhooks in `shopify.app.toml` (this repo already lists several).

---

## Resources

- [Shopify app intro](https://shopify.dev/docs/apps/getting-started)
- [Shopify CLI](https://shopify.dev/docs/apps/tools/cli)
- [Polaris web components](https://shopify.dev/docs/api/app-home/polaris-web-components)
- [Theme app extensions](https://shopify.dev/docs/apps/app-extensions/list)

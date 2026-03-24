# Prisma on an existing PostgreSQL database (P3005)

If deploy logs show:

```text
Error: P3005 — The database schema is not empty
```

Postgres already has tables (e.g. Python + pgvector) and Prisma has never recorded a migration. Do this **once**:

**Shortcut (from `loom-app/`):** copy `DATABASE_URL` from Railway Postgres → **Variables** (same string the app uses), then:

```bash
export DATABASE_URL='postgresql://...'
./scripts/railway-baseline.sh
```

That runs both migration SQL files in order (legacy `shopify` init, then move to `public."Session"`), then marks both migrations as applied.

Or step by step:

1. **Apply SQL** — run the contents of these files in order in Railway Query or `psql`:
   - `prisma/migrations/20260324120000_init_postgres_session/migration.sql`
   - `prisma/migrations/20260324161000_move_session_to_public/migration.sql`

2. **Mark migrations as applied** (does not run SQL again):

   ```bash
   cd loom-app
   DATABASE_URL="postgresql://..." npx prisma migrate resolve --applied 20260324120000_init_postgres_session
   DATABASE_URL="postgresql://..." npx prisma migrate resolve --applied 20260324161000_move_session_to_public
   ```

   Or: `npm run migrate:resolve-baseline` with `DATABASE_URL` set.

3. Redeploy the app — `prisma migrate deploy` should report **No pending migrations**.

`Session` is stored as **`public."Session"`** so it matches Railway’s default connection schema and the Prisma session adapter.

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

Or step by step:

1. **Apply the Session migration SQL** (creates schema `shopify` and `shopify."Session"`):

   - Railway: open your Postgres service → **Query** / **Data** → run the contents of  
     `prisma/migrations/20260324120000_init_postgres_session/migration.sql`

   - Or from your machine:

     ```bash
     psql "$DATABASE_URL" -f prisma/migrations/20260324120000_init_postgres_session/migration.sql
     ```

2. **Mark the migration as applied** (does not run SQL again):

   ```bash
   cd loom-app
   DATABASE_URL="postgresql://..." npx prisma migrate resolve --applied 20260324120000_init_postgres_session
   ```

   Use the **same** `DATABASE_URL` as Railway (loom-app service).

3. Redeploy the app — `prisma migrate deploy` should report **No pending migrations**.

`Session` lives in the **`shopify`** schema so it does not clash with tables in **`public`**.

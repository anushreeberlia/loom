#!/usr/bin/env bash
# One-shot: apply Session migration SQL + mark it applied (fixes P3005 on non-empty DB).
# Usage: DATABASE_URL="$(pbpaste)" ./scripts/railway-baseline.sh
#    or: export DATABASE_URL from Railway Postgres → Variables / Connect
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "Set DATABASE_URL to your Railway Postgres connection string." >&2
  exit 1
fi
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f prisma/migrations/20260324120000_init_postgres_session/migration.sql
npx prisma migrate resolve --applied 20260324120000_init_postgres_session
echo "Done. Redeploy the loom-app service."

#!/usr/bin/env bash
# One-shot for P3005: apply both migration SQL files + mark them applied (non-empty DB, no history).
# Usage: export DATABASE_URL='...' && ./scripts/railway-baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "Set DATABASE_URL to your Railway Postgres connection string." >&2
  exit 1
fi
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f prisma/migrations/20260324120000_init_postgres_session/migration.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f prisma/migrations/20260324161000_move_session_to_public/migration.sql
npx prisma migrate resolve --applied 20260324120000_init_postgres_session
npx prisma migrate resolve --applied 20260324161000_move_session_to_public
echo "Done. Redeploy the loom-app service."

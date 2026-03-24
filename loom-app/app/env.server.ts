let productionEnvAsserted = false;

/**
 * Fail fast on boot in production when required env is missing (clearer than a generic Shopify error).
 * Idempotent — safe to call from multiple entrypoints (e.g. db vs shopify).
 */
export function assertProductionEnv(): void {
  if (productionEnvAsserted) return;
  if (process.env.NODE_ENV !== "production") return;
  productionEnvAsserted = true;

  const required = [
    "SHOPIFY_API_KEY",
    "SHOPIFY_API_SECRET",
    "SCOPES",
    "SHOPIFY_APP_URL",
    "DATABASE_URL",
    "LOOM_BACKEND_URL",
  ] as const;

  const missing: string[] = [];
  for (const key of required) {
    if (!process.env[key]?.trim()) missing.push(key);
  }
  if (missing.length > 0) {
    throw new Error(
      `[loom-app] Missing required environment variables in production: ${missing.join(", ")}`,
    );
  }

  const db = process.env.DATABASE_URL!.trim();
  const isPostgresLike =
    /^postgres(ql)?:\/\//i.test(db) ||
    /^prisma(\+postgres)?:\/\//i.test(db);
  if (!isPostgresLike) {
    throw new Error(
      "[loom-app] DATABASE_URL must point to PostgreSQL (or Prisma Accelerate prisma+postgres://) in production",
    );
  }
}

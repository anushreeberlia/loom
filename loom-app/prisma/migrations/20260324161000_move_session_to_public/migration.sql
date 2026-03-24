-- Shopify session storage must use public."Session" so Prisma Client matches Railway's
-- default connection schema (schema=public) and the session storage adapter.
CREATE TABLE IF NOT EXISTS "public"."Session" (
    "id" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "isOnline" BOOLEAN NOT NULL DEFAULT false,
    "scope" TEXT,
    "expires" TIMESTAMP(3),
    "accessToken" TEXT NOT NULL,
    "userId" BIGINT,
    "firstName" TEXT,
    "lastName" TEXT,
    "email" TEXT,
    "accountOwner" BOOLEAN NOT NULL DEFAULT false,
    "locale" TEXT,
    "collaborator" BOOLEAN DEFAULT false,
    "emailVerified" BOOLEAN DEFAULT false,
    "refreshToken" TEXT,
    "refreshTokenExpires" TIMESTAMP(3),
    CONSTRAINT "Session_pkey" PRIMARY KEY ("id")
);

-- Prior migration created shopify."Session"; copy then drop so we don't lose OAuth rows.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'shopify' AND table_name = 'Session'
  ) THEN
    INSERT INTO "public"."Session"
    SELECT * FROM "shopify"."Session"
    ON CONFLICT ("id") DO NOTHING;
    DROP TABLE IF EXISTS "shopify"."Session";
  END IF;
END $$;

DROP SCHEMA IF EXISTS "shopify" CASCADE;

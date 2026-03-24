import "@shopify/shopify-app-react-router/adapters/node";
import {
  ApiVersion,
  AppDistribution,
  LogSeverity,
  shopifyApp,
} from "@shopify/shopify-app-react-router/server";
import { PrismaSessionStorage } from "@shopify/shopify-app-session-storage-prisma";
import prisma from "./db.server";

/** Public origin for OAuth / embedded app. Prefer explicit SHOPIFY_APP_URL in production. */
function shopifyAppUrl(): string {
  const explicit = process.env.SHOPIFY_APP_URL?.trim();
  if (explicit) return explicit.replace(/\/$/, "");
  const railwayStatic = process.env.RAILWAY_STATIC_URL?.trim();
  if (railwayStatic) return railwayStatic.replace(/\/$/, "");
  const railwayHost = process.env.RAILWAY_PUBLIC_DOMAIN?.trim();
  if (railwayHost) return `https://${railwayHost.replace(/\/$/, "")}`;
  return "";
}

const shopify = shopifyApp({
  apiKey: process.env.SHOPIFY_API_KEY,
  apiSecretKey: process.env.SHOPIFY_API_SECRET || "",
  apiVersion: ApiVersion.October25,
  scopes: process.env.SCOPES?.split(","),
  appUrl: shopifyAppUrl(),
  authPathPrefix: "/auth",
  sessionStorage: new PrismaSessionStorage(prisma),
  distribution: AppDistribution.AppStore,
  future: {
    // Dev: token refresh can call Shopify and appear to hang; production keeps expiring tokens.
    expiringOfflineAccessTokens: process.env.NODE_ENV === "production",
  },
  ...(process.env.NODE_ENV !== "production"
    ? {
        // Surfaces real auth/session errors in the terminal; "Handling response" in Admin often hides them.
        logger: { level: LogSeverity.Debug, timestamps: true },
      }
    : {}),
  ...(process.env.SHOP_CUSTOM_DOMAIN
    ? { customShopDomains: [process.env.SHOP_CUSTOM_DOMAIN] }
    : {}),
});

export default shopify;
export const apiVersion = ApiVersion.October25;
export const addDocumentResponseHeaders = shopify.addDocumentResponseHeaders;
export const authenticate = shopify.authenticate;
export const unauthenticated = shopify.unauthenticated;
export const login = shopify.login;
export const registerWebhooks = shopify.registerWebhooks;
export const sessionStorage = shopify.sessionStorage;

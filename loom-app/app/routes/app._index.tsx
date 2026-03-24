import { useEffect, useMemo } from "react";
import type { HeadersFunction, ActionFunctionArgs } from "react-router";
import { useFetcher, useOutletContext } from "react-router";
import { useAppBridge } from "@shopify/app-bridge-react";
import { authenticate } from "../shopify.server";
import { getLoomBackendUrl } from "../loomBackend.server";
import { loomError, loomLog } from "../loomLog";
import { boundary } from "@shopify/shopify-app-react-router/server";

const BACKEND_FETCH_MS = 8_000;

type CatalogStatus = {
  product_count: number;
  outfit_count: number;
  synced_at: string | null;
  pending_processing: number;
  recent_products?: { name: string; shopify_product_id: string; product_url: string | null }[];
};

const DEFAULT_STATUS: CatalogStatus = {
  product_count: 0,
  outfit_count: 0,
  synced_at: null,
  pending_processing: 0,
  recent_products: [],
};

async function fetchWithTimeout(
  url: string,
  init: RequestInit = {},
  timeoutMs = BACKEND_FETCH_MS,
): Promise<Response> {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(id);
  }
}

// ── Action: Loom calls run here after the shell loads (bootstrap + sync) ─────

export const action = async ({ request }: ActionFunctionArgs) => {
  const t0 = Date.now();
  const { session } = await authenticate.admin(request);
  const loomBackend = getLoomBackendUrl();
  const { shop, accessToken } = session;
  const formData = await request.formData();
  const intent = String(formData.get("intent") ?? "sync");

  loomLog("app._index.action", "start", {
    intent,
    shop,
    msAuth: Date.now() - t0,
  });

  if (intent === "bootstrap") {
    let backendUnreachable = false;
    let status: CatalogStatus = { ...DEFAULT_STATUS };

    const tInstall = Date.now();
    try {
      const res = await fetchWithTimeout(`${loomBackend}/shopify/install`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          shop_domain: shop,
          access_token: accessToken,
          scope: session.scope ?? "",
        }),
      });
      loomLog("app._index.action", "POST /shopify/install", {
        ok: res.ok,
        status: res.status,
        ms: Date.now() - tInstall,
      });
      if (!res.ok) backendUnreachable = true;
    } catch (e) {
      backendUnreachable = true;
      loomError("app._index.action", "POST /shopify/install failed", e);
    }

    const tStatus = Date.now();
    try {
      const res = await fetchWithTimeout(
        `${loomBackend}/shopify/catalog/status?shop_domain=${encodeURIComponent(shop)}`,
      );
      if (res.ok) status = await res.json();
      else backendUnreachable = true;
      loomLog("app._index.action", "GET /shopify/catalog/status", {
        ok: res.ok,
        status: res.status,
        ms: Date.now() - tStatus,
      });
    } catch (e) {
      backendUnreachable = true;
      loomError("app._index.action", "GET /shopify/catalog/status failed", e);
    }

    loomLog("app._index.action", "bootstrap done", {
      backendUnreachable,
      totalMs: Date.now() - t0,
      products: status.product_count,
    });

    return {
      intent: "bootstrap" as const,
      status,
      backendUnreachable,
      loomBackendUrl: loomBackend,
    };
  }

  const tSync = Date.now();
  try {
    const res = await fetchWithTimeout(`${loomBackend}/shopify/catalog/sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shop_domain: shop, access_token: accessToken }),
    });
    loomLog("app._index.action", "POST /shopify/catalog/sync", {
      ok: res.ok,
      status: res.status,
      ms: Date.now() - tSync,
    });
    if (!res.ok) loomError("app._index.action", "catalog/sync non-OK", res.status);
  } catch (e) {
    loomError("app._index.action", "catalog/sync request failed", e);
  }

  loomLog("app._index.action", "sync intent done", { totalMs: Date.now() - t0 });

  return { intent: "sync" as const, syncing: true };
};

// ── UI ────────────────────────────────────────────────────────────────────────

type AppOutletContext = { shop: string; loomBackendUrl: string };

function adminProductUrl(shop: string, shopifyGid: string): string | null {
  const handle = shop.replace(/\.myshopify\.com$/i, "");
  const id = shopifyGid.includes("/") ? shopifyGid.split("/").pop() : shopifyGid;
  if (!handle || !id) return null;
  return `https://admin.shopify.com/store/${handle}/products/${id}`;
}

export default function Index() {
  const { shop, loomBackendUrl } = useOutletContext<AppOutletContext>();
  const bootstrap = useFetcher<typeof action>();
  const syncFetcher = useFetcher<typeof action>();
  const shopify = useAppBridge();

  useEffect(() => {
    if (import.meta.env.DEV) {
      console.log("[loom-app] [client] submitting bootstrap action");
    }
    bootstrap.submit({ intent: "bootstrap" }, { method: "POST" });
    // One-time bootstrap after shell load; install + status stay on the server.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const status = useMemo(() => {
    if (bootstrap.data?.intent === "bootstrap") return bootstrap.data.status;
    return DEFAULT_STATUS;
  }, [bootstrap.data]);

  const backendUnreachable =
    bootstrap.data?.intent === "bootstrap" ? bootstrap.data.backendUnreachable : false;

  const catalogLoading =
    (bootstrap.state === "loading" || bootstrap.state === "submitting") && !bootstrap.data;

  const isSyncing =
    status.pending_processing > 0 ||
    (["loading", "submitting"].includes(syncFetcher.state) &&
      syncFetcher.formMethod === "POST");

  useEffect(() => {
    if (syncFetcher.data?.intent === "sync" && syncFetcher.data.syncing) {
      shopify.toast.show("Catalog sync started — this may take a few minutes.");
    }
  }, [syncFetcher.data, shopify]);

  const triggerSync = () =>
    syncFetcher.submit({ intent: "sync" }, { method: "POST" });

  return (
    <s-page heading="Loom — AI Outfit Generation">
      {catalogLoading && (
        <s-section heading="Catalog">
          <s-paragraph>Connecting to Loom API…</s-paragraph>
        </s-section>
      )}
      {backendUnreachable && (
        <s-section heading="Backend connection">
          <s-paragraph>
            Could not reach the Loom API at {loomBackendUrl} within a few seconds. Check
            LOOM_BACKEND_URL in hosting env and that {loomBackendUrl}/shopify/health responds,
            then reload.
          </s-paragraph>
        </s-section>
      )}
      <s-button
        slot="primary-action"
        onClick={triggerSync}
        {...(isSyncing ? { loading: true } : {})}
      >
        {isSyncing ? "Syncing..." : "Sync Catalog"}
      </s-button>

      <s-section heading="Catalog Status">
        <s-stack direction="inline" gap="base">
          <s-box padding="base" borderWidth="base" borderRadius="base">
            <s-heading>{status.product_count}</s-heading>
            <s-paragraph>Products processed</s-paragraph>
          </s-box>
          <s-box padding="base" borderWidth="base" borderRadius="base">
            <s-heading>{status.outfit_count}</s-heading>
            <s-paragraph>Products with outfits</s-paragraph>
          </s-box>
          {status.pending_processing > 0 && (
            <s-box padding="base" borderWidth="base" borderRadius="base">
              <s-heading>{status.pending_processing}</s-heading>
              <s-paragraph>Pending processing</s-paragraph>
            </s-box>
          )}
        </s-stack>
        {status.synced_at && (
          <s-paragraph>
            Last synced: {new Date(status.synced_at).toLocaleString()}
          </s-paragraph>
        )}
      </s-section>

      {(status.recent_products?.length ?? 0) > 0 && (
        <s-section heading="Sample products in Loom">
          <s-paragraph>
            These are the latest items processed in your catalog (for testing in Admin). Shoppers only
            see outfits after you add the theme block (Setup step 2).
          </s-paragraph>
          <s-unordered-list>
            {(status.recent_products ?? []).map((p) => {
              const adminUrl = adminProductUrl(shop, p.shopify_product_id);
              return (
                <s-list-item key={p.shopify_product_id}>
                  {adminUrl ? (
                    <s-link href={adminUrl} target="_blank">
                      {p.name || p.shopify_product_id}
                    </s-link>
                  ) : (
                    <span>{p.name || p.shopify_product_id}</span>
                  )}
                  {p.product_url ? (
                    <>
                      {" · "}
                      <s-link href={p.product_url} target="_blank">
                        Storefront
                      </s-link>
                    </>
                  ) : null}
                </s-list-item>
              );
            })}
          </s-unordered-list>
        </s-section>
      )}

      <s-section heading="How it works">
        <s-paragraph>
          Loom analyzes every product in your catalog and automatically generates complete outfit
          suggestions. These appear as a &quot;Shop the Look&quot; block on your product pages,
          showing shoppers how to style each item with other pieces from your store.
        </s-paragraph>
        <s-unordered-list>
          <s-list-item>Every product gets multiple outfit directions (Classic, Trendy, Bold)</s-list-item>
          <s-list-item>Outfits update automatically when you add new products</s-list-item>
          <s-list-item>Only shows in-stock items from your catalog</s-list-item>
          <s-list-item>Personalized per shopper based on their preferences</s-list-item>
        </s-unordered-list>
      </s-section>

      <s-section slot="aside" heading="Setup">
        <s-paragraph>
          <s-text type="strong">Step 1</s-text> — Click &quot;Sync Catalog&quot; to process your
          products. This runs in the background.
        </s-paragraph>
        <s-paragraph>
          <s-text type="strong">Step 2</s-text> — Add the &quot;Shop the Look by Loom&quot; block
          to your product pages in the theme editor.
        </s-paragraph>
        <s-paragraph>
          <s-text type="strong">Step 3</s-text> — Shoppers will see AI-generated outfit
          suggestions on every product page.
        </s-paragraph>
      </s-section>

      <s-section slot="aside" heading="Store">
        <s-paragraph>{shop}</s-paragraph>
      </s-section>
    </s-page>
  );
}

export const headers: HeadersFunction = (headersArgs) => {
  return boundary.headers(headersArgs);
};

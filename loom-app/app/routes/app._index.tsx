import { useEffect, useMemo } from "react";
import type { HeadersFunction, LoaderFunctionArgs, ActionFunctionArgs } from "react-router";
import { useFetcher, useLoaderData } from "react-router";
import { useAppBridge } from "@shopify/app-bridge-react";
import { authenticate } from "../shopify.server";
import { boundary } from "@shopify/shopify-app-react-router/server";

const LOOM_BACKEND = process.env.LOOM_BACKEND_URL || "http://127.0.0.1:8001";

const BACKEND_FETCH_MS = 8_000;

const DEFAULT_STATUS = {
  product_count: 0,
  outfit_count: 0,
  synced_at: null as string | null,
  pending_processing: 0,
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

// ── Loader: auth only — no external HTTP, so the iframe gets HTML immediately ─

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const { session } = await authenticate.admin(request);
  return { shop: session.shop, loomBackendUrl: LOOM_BACKEND };
};

// ── Action: Loom calls run here after the shell loads (bootstrap + sync) ─────

export const action = async ({ request }: ActionFunctionArgs) => {
  const { session } = await authenticate.admin(request);
  const { shop, accessToken } = session;
  const formData = await request.formData();
  const intent = String(formData.get("intent") ?? "sync");

  if (intent === "bootstrap") {
    let backendUnreachable = false;
    let status = { ...DEFAULT_STATUS };

    try {
      const res = await fetchWithTimeout(`${LOOM_BACKEND}/shopify/install`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          shop_domain: shop,
          access_token: accessToken,
          scope: session.scope ?? "",
        }),
      });
      if (!res.ok) backendUnreachable = true;
    } catch (e) {
      backendUnreachable = true;
      console.error("Failed to register install with Loom backend:", e);
    }

    try {
      const res = await fetchWithTimeout(
        `${LOOM_BACKEND}/shopify/catalog/status?shop_domain=${encodeURIComponent(shop)}`,
      );
      if (res.ok) status = await res.json();
      else backendUnreachable = true;
    } catch (e) {
      backendUnreachable = true;
      console.error("Failed to fetch status:", e);
    }

    return {
      intent: "bootstrap" as const,
      status,
      backendUnreachable,
      loomBackendUrl: LOOM_BACKEND,
    };
  }

  try {
    const res = await fetchWithTimeout(`${LOOM_BACKEND}/shopify/catalog/sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shop_domain: shop, access_token: accessToken }),
    });
    if (!res.ok) console.error("catalog/sync failed:", res.status);
  } catch (e) {
    console.error("catalog/sync request failed:", e);
  }

  return { intent: "sync" as const, syncing: true };
};

// ── UI ────────────────────────────────────────────────────────────────────────

export default function Index() {
  const { shop, loomBackendUrl } = useLoaderData<typeof loader>();
  const bootstrap = useFetcher<typeof action>();
  const syncFetcher = useFetcher<typeof action>();
  const shopify = useAppBridge();

  useEffect(() => {
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
            LOOM_BACKEND_URL and that https://loom-style.com/shopify/health responds, then
            reload.
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

import { useEffect } from "react";
import type { HeadersFunction, LoaderFunctionArgs, ActionFunctionArgs } from "react-router";
import { useFetcher, useLoaderData } from "react-router";
import { useAppBridge } from "@shopify/app-bridge-react";
import { authenticate } from "../shopify.server";
import { boundary } from "@shopify/shopify-app-react-router/server";

const LOOM_BACKEND = process.env.LOOM_BACKEND_URL || "http://127.0.0.1:8001";

// ── Loader: authenticate + register install + fetch status ───────────────────

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const { session } = await authenticate.admin(request);
  const { shop, accessToken } = session;

  // Register install with Loom backend (idempotent)
  try {
    await fetch(`${LOOM_BACKEND}/shopify/install`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        shop_domain: shop,
        access_token: accessToken,
        scope: "",
      }),
    });
  } catch (e) {
    console.error("Failed to register install with Loom backend:", e);
  }

  // Fetch catalog status
  let status = { product_count: 0, outfit_count: 0, synced_at: null, pending_processing: 0 };
  try {
    const res = await fetch(`${LOOM_BACKEND}/shopify/catalog/status?shop_domain=${shop}`);
    if (res.ok) status = await res.json();
  } catch (e) {
    console.error("Failed to fetch status:", e);
  }

  return { shop, status };
};

// ── Action: manual re-sync ────────────────────────────────────────────────────

export const action = async ({ request }: ActionFunctionArgs) => {
  const { session } = await authenticate.admin(request);
  const { shop, accessToken } = session;

  await fetch(`${LOOM_BACKEND}/shopify/catalog/sync`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ shop_domain: shop, access_token: accessToken }),
  });

  return { syncing: true };
};

// ── UI ────────────────────────────────────────────────────────────────────────

export default function Index() {
  const { shop, status } = useLoaderData<typeof loader>();
  const fetcher = useFetcher<typeof action>();
  const shopify = useAppBridge();

  const isSyncing =
    status.pending_processing > 0 ||
    (["loading", "submitting"].includes(fetcher.state) && fetcher.formMethod === "POST");

  useEffect(() => {
    if (fetcher.data?.syncing) {
      shopify.toast.show("Catalog sync started — this may take a few minutes.");
    }
  }, [fetcher.data?.syncing, shopify]);

  const triggerSync = () => fetcher.submit({}, { method: "POST" });

  return (
    <s-page heading="Loom — AI Outfit Generation">
      <s-button
        slot="primary-action"
        onClick={triggerSync}
        {...(isSyncing ? { loading: true } : {})}
      >
        {isSyncing ? "Syncing..." : "Sync Catalog"}
      </s-button>

      <s-section heading="Catalog Status">
        <s-stack direction="inline" gap="loose">
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
          Loom analyzes every product in your catalog and automatically generates
          complete outfit suggestions. These appear as a &quot;Shop the Look&quot; block
          on your product pages, showing shoppers how to style each item with
          other pieces from your store.
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
          <s-text fontWeight="bold">Step 1</s-text> — Click &quot;Sync Catalog&quot; to process
          your products. This runs in the background.
        </s-paragraph>
        <s-paragraph>
          <s-text fontWeight="bold">Step 2</s-text> — Add the &quot;Shop the Look by Loom&quot;
          block to your product pages in the theme editor.
        </s-paragraph>
        <s-paragraph>
          <s-text fontWeight="bold">Step 3</s-text> — Shoppers will see AI-generated
          outfit suggestions on every product page.
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

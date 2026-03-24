import type { ActionFunctionArgs } from "react-router";
import { authenticate } from "../shopify.server";
import db from "../db.server";
import { getLoomBackendUrl } from "../loomBackend.server";

export const action = async ({ request }: ActionFunctionArgs) => {
  const { shop, session, topic } = await authenticate.webhook(request);

  console.log(`Received ${topic} webhook for ${shop}`);

  // Webhook requests can trigger multiple times and after an app has already been uninstalled.
  // If this webhook already ran, the session may have been deleted previously.
  if (session) {
    await db.session.deleteMany({ where: { shop } });
  }

  try {
    const loomBackend = getLoomBackendUrl();
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const shared = process.env.LOOM_BACKEND_SHARED_SECRET;
    if (shared) headers["Authorization"] = `Bearer ${shared}`;
    await fetch(`${loomBackend}/shopify/notify-uninstall`, {
      method: "POST",
      headers,
      body: JSON.stringify({ shop_domain: shop }),
    });
  } catch (e) {
    console.error("Failed to notify Loom backend of uninstall:", e);
  }

  return new Response();
};

import type { ActionFunctionArgs } from "react-router";
import { authenticate } from "../shopify.server";

const LOOM_BACKEND = process.env.LOOM_BACKEND_URL || "http://127.0.0.1:8001";

export const action = async ({ request }: ActionFunctionArgs) => {
  const { shop, topic, payload } = await authenticate.webhook(request);
  console.log(`Received ${topic} webhook for ${shop}`);

  try {
    const body = JSON.stringify(payload);
    await fetch(`${LOOM_BACKEND}/shopify/webhooks/product_created`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Shopify-Shop-Domain": shop,
      },
      body,
    });
  } catch (e) {
    console.error(`Failed to forward ${topic} to Loom backend:`, e);
  }

  return new Response();
};

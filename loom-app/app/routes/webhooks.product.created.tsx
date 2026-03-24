import type { ActionFunctionArgs } from "react-router";
import { authenticate } from "../shopify.server";
import { getLoomBackendUrl } from "../loomBackend.server";

export const action = async ({ request }: ActionFunctionArgs) => {
  const { shop, topic, payload } = await authenticate.webhook(request);
  console.log(`Received ${topic} webhook for ${shop}`);

  try {
    const loomBackend = getLoomBackendUrl();
    const body = JSON.stringify(payload);
    await fetch(`${loomBackend}/shopify/webhooks/product_created`, {
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

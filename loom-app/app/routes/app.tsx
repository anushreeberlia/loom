import type { HeadersFunction, LoaderFunctionArgs } from "react-router";
import { Outlet, useLoaderData, useRouteError } from "react-router";
import { boundary } from "@shopify/shopify-app-react-router/server";
import { AppProvider } from "@shopify/shopify-app-react-router/react";

import { authenticate } from "../shopify.server";
import { getLoomBackendUrl } from "../loomBackend.server";
import { loomLog } from "../loomLog";

/** Single auth + session read per /app request (avoids parallel loaders both hitting Prisma/SQLite). */
export const loader = async ({ request }: LoaderFunctionArgs) => {
  const t0 = Date.now();
  const path = new URL(request.url).pathname;
  loomLog("app.loader", "authenticate.admin start", { path });

  const { session } = await authenticate.admin(request);
  const loomBackendUrl = getLoomBackendUrl();

  loomLog("app.loader", "authenticate.admin ok", {
    ms: Date.now() - t0,
    shop: session.shop,
    loomBackendHost: new URL(loomBackendUrl).hostname,
  });

  return {
    // eslint-disable-next-line no-undef
    apiKey: process.env.SHOPIFY_API_KEY || "",
    shop: session.shop,
    loomBackendUrl,
  };
};

export default function App() {
  const { apiKey, shop, loomBackendUrl } = useLoaderData<typeof loader>();

  return (
    <AppProvider embedded apiKey={apiKey}>
      <s-app-nav>
        <s-link href="/app">Home</s-link>
        <s-link href="/app/additional">Additional page</s-link>
      </s-app-nav>
      <Outlet context={{ shop, loomBackendUrl }} />
    </AppProvider>
  );
}

// Shopify needs React Router to catch some thrown responses, so that their headers are included in the response.
export function ErrorBoundary() {
  return boundary.error(useRouteError());
}

export const headers: HeadersFunction = (headersArgs) => {
  return boundary.headers(headersArgs);
};

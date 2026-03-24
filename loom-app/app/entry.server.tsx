import { PassThrough } from "stream";
import { renderToPipeableStream } from "react-dom/server";
import { ServerRouter } from "react-router";
import { createReadableStreamFromReadable } from "@react-router/node";
import { type EntryContext } from "react-router";
import { isbot } from "isbot";
import { addDocumentResponseHeaders } from "./shopify.server";

/** Shell streaming budget (ms); dev tunnels / cold starts can be slow. */
export const streamTimeout = 25000;

export default async function handleRequest(
  request: Request,
  responseStatusCode: number,
  responseHeaders: Headers,
  reactRouterContext: EntryContext
) {
  if (process.env.NODE_ENV === "development") {
    const u = new URL(request.url);
    if (
      u.pathname.startsWith("/app") ||
      u.pathname.startsWith("/auth") ||
      u.pathname.startsWith("/webhooks")
    ) {
      console.log(
        `[loom-app] [http] ${request.method} ${u.pathname}${u.search ? u.search : ""}`,
      );
    }
  }

  addDocumentResponseHeaders(request, responseHeaders);
  // Shopify's helper only sets frame-ancestors when `shop` is in the query. Some embedded
  // navigations send `embedded=1` without `shop` yet — ensure Admin can still frame the app.
  const reqUrl = new URL(request.url);
  if (
    reqUrl.searchParams.get("embedded") === "1" &&
    !reqUrl.searchParams.get("shop") &&
    !responseHeaders.get("Content-Security-Policy")
  ) {
    responseHeaders.set(
      "Content-Security-Policy",
      "frame-ancestors https://admin.shopify.com https://*.spin.dev https://admin.myshopify.io https://admin.shop.dev;",
    );
  }
  const userAgent = request.headers.get("user-agent");
  const callbackName = isbot(userAgent ?? '')
    ? "onAllReady"
    : "onShellReady";

  return new Promise((resolve, reject) => {
    const { pipe, abort } = renderToPipeableStream(
      <ServerRouter
        context={reactRouterContext}
        url={request.url}
      />,
      {
        [callbackName]: () => {
          const body = new PassThrough();
          const stream = createReadableStreamFromReadable(body);

          responseHeaders.set("Content-Type", "text/html");
          resolve(
            new Response(stream, {
              headers: responseHeaders,
              status: responseStatusCode,
            })
          );
          pipe(body);
        },
        onShellError(error) {
          reject(error);
        },
        onError(error) {
          responseStatusCode = 500;
          console.error(error);
        },
      }
    );

    // Automatically timeout the React renderer after 6 seconds, which ensures
    // React has enough time to flush down the rejected boundary contents
    setTimeout(abort, streamTimeout + 1000);
  });
}

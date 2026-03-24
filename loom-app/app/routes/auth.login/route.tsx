import { AppProvider } from "@shopify/shopify-app-react-router/react";
import type { ActionFunctionArgs, LoaderFunctionArgs } from "react-router";
import { useFetcher, useLoaderData } from "react-router";

import { login } from "../../shopify.server";
import { normalizeShopDomain } from "../../utils/shopDomain.server";
import { loginErrorMessage } from "./error.server";

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const url = new URL(request.url);
  // Avoid "Missing shop" on first paint when opening /auth/login directly (no ?shop=).
  if (request.method === "GET" && !url.searchParams.get("shop")) {
    return { errors: {} };
  }

  const errors = loginErrorMessage(await login(request));

  return { errors };
};

export const action = async ({ request }: ActionFunctionArgs) => {
  const incoming = await request.formData();
  const raw = incoming.get("shop");
  const fd = new FormData();
  if (typeof raw === "string" && raw.trim()) {
    fd.set("shop", normalizeShopDomain(raw));
  }
  // Do not forward request.headers: old Content-Type / Content-Length breaks parsing when body
  // is a new FormData(), so login()'s request.formData() returns empty → Missing shop (see
  // @shopify/shopify-app-react-router authenticate/login/login.js).
  const loginRequest = new Request(request.url, { method: "POST", body: fd });
  const errors = loginErrorMessage(await login(loginRequest));

  return {
    errors,
  };
};

export default function Auth() {
  const loaderData = useLoaderData<typeof loader>();
  const fetcher = useFetcher<typeof action>();
  const { errors } = fetcher.data ?? loaderData;

  return (
    <AppProvider embedded={false}>
      <div
        style={{
          padding: "1.5rem",
          maxWidth: "26rem",
          margin: "2rem auto",
          border: "1px solid #e3e3e3",
          borderRadius: "12px",
          background: "#fff",
        }}
      >
        {/*
          Explicit FormData via fetcher.submit: RR <Form> was still POSTing without `shop` in
          some environments (MissingShop despite a filled input).
        */}
        <fetcher.Form
          method="post"
          onSubmit={(e) => {
            e.preventDefault();
            const shopEl = e.currentTarget.elements.namedItem(
              "shop",
            ) as HTMLInputElement | null;
            const raw = (shopEl?.value ?? "").trim();
            const fd = new FormData();
            fd.set("shop", raw);
            fetcher.submit(fd, { method: "post" });
          }}
        >
          <h2 style={{ margin: "0 0 1rem", fontSize: "1.25rem", fontWeight: 600 }}>Log in</h2>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0.5rem",
              marginBottom: "1rem",
            }}
          >
            <label htmlFor="shop-login" style={{ fontWeight: 600 }}>
              Shop domain
            </label>
            <input
              id="shop-login"
              name="shop"
              type="text"
              autoComplete="url"
              placeholder="loom-10146.myshopify.com"
              style={{
                padding: "0.5rem 0.75rem",
                border: errors.shop ? "2px solid #c5280c" : "1px solid #ccc",
                borderRadius: "6px",
                fontSize: "1rem",
              }}
              aria-invalid={Boolean(errors.shop)}
              aria-describedby={errors.shop ? "shop-login-error" : undefined}
            />
            {errors.shop ? (
              <p
                id="shop-login-error"
                style={{ color: "#c5280c", margin: 0, fontSize: "0.875rem" }}
              >
                {errors.shop}
              </p>
            ) : null}
            <p style={{ margin: 0, fontSize: "0.8125rem", color: "#666" }}>
              Use your .myshopify.com domain (short name like &quot;loom-10146&quot; works).
            </p>
          </div>
          <button
            type="submit"
            disabled={fetcher.state !== "idle"}
            style={{
              padding: "0.5rem 1rem",
              fontSize: "0.9375rem",
              fontWeight: 600,
              borderRadius: "6px",
              border: "1px solid #ccc",
              background: "#f6f6f7",
              cursor: fetcher.state === "idle" ? "pointer" : "wait",
            }}
          >
            {fetcher.state !== "idle" ? "Signing in…" : "Log in"}
          </button>
        </fetcher.Form>
      </div>
    </AppProvider>
  );
}

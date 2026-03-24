import type { LoaderFunctionArgs } from "react-router";
import { redirect, Form, useLoaderData } from "react-router";

import { login } from "../../shopify.server";

import styles from "./styles.module.css";

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const url = new URL(request.url);

  if (url.searchParams.get("shop")) {
    throw redirect(`/app?${url.searchParams.toString()}`);
  }

  return { showForm: Boolean(login) };
};

const features = [
  {
    title: "Visual search",
    body: "Shoppers find similar styles from your catalog—powered by your product imagery.",
  },
  {
    title: "Shop the Look",
    body: "Surface complete outfits on the storefront so customers buy the full look.",
  },
  {
    title: "Your catalog, synced",
    body: "Connect Loom to your store and keep recommendations aligned with what you sell.",
  },
] as const;

export default function App() {
  const { showForm } = useLoaderData<typeof loader>();

  return (
    <div className={styles.page}>
      <div className={styles.glow} aria-hidden />
      <main className={styles.main}>
        <p className={styles.eyebrow}>Shopify app</p>
        <h1 className={styles.heading}>Loom</h1>
        <p className={styles.lead}>
          Style discovery for your store—visual search and curated looks that turn
          browsers into buyers.
        </p>

        {showForm && (
          <div className={styles.card}>
            <h2 className={styles.cardTitle}>Sign in to your store</h2>
            <p className={styles.cardHint}>
              Use your store&apos;s myshopify.com domain to open the app in the
              Shopify admin.
            </p>
            <Form className={styles.form} method="post" action="/auth/login">
              <div className={styles.field}>
                <label className={styles.label} htmlFor="shop-domain">
                  Store domain
                </label>
                <input
                  id="shop-domain"
                  className={styles.input}
                  type="text"
                  name="shop"
                  placeholder="your-store.myshopify.com"
                  autoComplete="on"
                  spellCheck={false}
                />
                <span className={styles.help}>
                  Example: <kbd className={styles.kbd}>loom-10146.myshopify.com</kbd> or
                  just <kbd className={styles.kbd}>loom</kbd>
                </span>
              </div>
              <button className={styles.button} type="submit">
                Continue to Shopify
              </button>
            </Form>
          </div>
        )}

        <ul className={styles.features}>
          {features.map((f) => (
            <li key={f.title} className={styles.feature}>
              <span className={styles.featureIcon} aria-hidden />
              <strong className={styles.featureTitle}>{f.title}</strong>
              <p className={styles.featureBody}>{f.body}</p>
            </li>
          ))}
        </ul>
      </main>
    </div>
  );
}

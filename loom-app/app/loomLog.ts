/** Prefix all Shopify app server logs for easy filtering: `npm run dev 2>&1 | grep loom-app` */

const P = "[loom-app]";

export function loomLog(
  scope: string,
  message: string,
  extra?: Record<string, string | number | boolean | null | undefined>,
): void {
  if (extra && Object.keys(extra).length > 0) {
    const bits = Object.entries(extra)
      .map(([k, v]) => `${k}=${v}`)
      .join(" ");
    console.log(`${P} [${scope}] ${message} | ${bits}`);
  } else {
    console.log(`${P} [${scope}] ${message}`);
  }
}

export function loomError(scope: string, message: string, err: unknown): void {
  console.error(`${P} [${scope}] ${message}`, err);
}

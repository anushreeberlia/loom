/**
 * Normalizes shop input for OAuth login.
 * "loom" → "loom.myshopify.com"; leaves full domains and URLs unchanged.
 */
export function normalizeShopDomain(input: string): string {
  let s = input.trim().toLowerCase();
  s = s.replace(/^https?:\/\//, "");
  const host = s.split("/")[0] ?? s;
  if (!host.includes(".")) {
    return `${host}.myshopify.com`;
  }
  return host;
}

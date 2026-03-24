/** Public Loom Python API origin (no trailing slash). */
export function getLoomBackendUrl(): string {
  const raw = process.env.LOOM_BACKEND_URL?.trim();
  if (raw) return raw.replace(/\/$/, "");

  if (process.env.NODE_ENV === "production") {
    throw new Error(
      "LOOM_BACKEND_URL is required in production (e.g. https://api.example.com)",
    );
  }

  return "http://127.0.0.1:8001";
}

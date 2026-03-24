import type { LoaderFunctionArgs } from "react-router";
import prisma from "../db.server";

/**
 * Liveness/readiness for Railway (HTTP health checks). Does not require Shopify auth.
 * Set health check path to `/health` in your host dashboard.
 */
export async function loader({ request }: LoaderFunctionArgs) {
  if (request.method !== "GET") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  try {
    await prisma.$queryRaw`SELECT 1`;
  } catch {
    return new Response("unhealthy: database", {
      status: 503,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }

  return new Response("ok", {
    status: 200,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

export default function Health() {
  return null;
}

import { PrismaClient } from "@prisma/client";
import { assertProductionEnv } from "./env.server";

assertProductionEnv();

declare global {
  // eslint-disable-next-line no-var
  var prismaGlobal: PrismaClient;
}

const prismaLog: ("query" | "error" | "warn")[] =
  process.env.NODE_ENV === "development" ? ["query", "error", "warn"] : ["error"];

if (process.env.NODE_ENV !== "production") {
  if (!global.prismaGlobal) {
    global.prismaGlobal = new PrismaClient({ log: prismaLog });
  }
}

const prisma = global.prismaGlobal ?? new PrismaClient({ log: prismaLog });

export default prisma;

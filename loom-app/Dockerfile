# Multi-stage: Vite + TypeScript live in devDependencies; production image only keeps runtime deps.
FROM node:20-alpine AS builder
RUN apk add --no-cache openssl
WORKDIR /app

COPY package.json package-lock.json* ./
COPY patches ./patches
RUN npm ci

COPY . .
RUN npx prisma generate
RUN npm run build

FROM node:20-alpine
RUN apk add --no-cache openssl
WORKDIR /app

ENV NODE_ENV=production
# HTTP server port (react-router-serve reads $PORT). On Railway use ~8080 or omit PORT.
# Never set PORT=5432 here or in Railway vars — 5432 is Postgres; the app will listen on the DB port and the site will break.
ENV PORT=8080

COPY package.json package-lock.json* ./
COPY patches ./patches
RUN npm ci --omit=dev && npm cache clean --force

COPY prisma ./prisma
RUN npx prisma generate

COPY --from=builder /app/build ./build

EXPOSE 8080

CMD ["npm", "run", "docker-start"]

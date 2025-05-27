# syntax = docker/dockerfile:1

FROM node:22-alpine AS base

LABEL fly_launch_runtime="Remix"

WORKDIR /app

ENV NODE_ENV="production"

ARG PNPM_VERSION=10.11.0
RUN npm install -g pnpm@$PNPM_VERSION

FROM base AS build

WORKDIR /app

COPY --link ./pnpm-lock.yaml ./
COPY --link ./pnpm-workspace.yaml ./
RUN pnpm install --prod=false

COPY --link ./www-artintellica-com ./www-artintellica-com

WORKDIR /app/www-artintellica-com

RUN pnpm install --prod=false

RUN pnpm run build

RUN pnpm prune --prod

FROM base

WORKDIR /app

COPY --from=build /app /app

WORKDIR /app/www-artintellica-com

EXPOSE 3000

CMD ["pnpm", "run", "start"]

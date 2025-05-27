FROM oven/bun:1 AS dependencies-env
COPY . /app

FROM dependencies-env AS development-dependencies-env
COPY ./package.json bun.lock /app/
WORKDIR /app
RUN bun install --frozen-lockfile

FROM dependencies-env AS production-dependencies-env
COPY ./package.json bun.lock /app/
WORKDIR /app
RUN bun install --production

FROM dependencies-env AS build-env
COPY ./package.json bun.lock /app/
COPY --from=development-dependencies-env /app/node_modules /app/node_modules
WORKDIR /app/www-artintellica-com
RUN bun run build

FROM dependencies-env
COPY ./package.json bun.lock /app/
COPY --from=production-dependencies-env /app/node_modules /app/node_modules
COPY --from=production-dependencies-env /app/www-artintellica-com/node_modules /app/www-artintellica-com/node_modules
COPY --from=build-env /app/www-artintellica-com/build /app/www-artintellica-com/build
WORKDIR /app/www-artintellica-com
CMD ["bun", "run", "start"]

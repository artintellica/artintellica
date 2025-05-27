import { reactRouter } from "@react-router/dev/vite";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";
import tailwindcss from "@tailwindcss/vite";
import { denyImports } from "vite-env-only";

export default defineConfig({
  plugins: [reactRouter(), tsconfigPaths(), tailwindcss(), denyImports({
    client: {
      files: ["**/server-only/**/*"],
    },
    server: {
      files: ["**/client-only/**/*"],
    },
  })],
});

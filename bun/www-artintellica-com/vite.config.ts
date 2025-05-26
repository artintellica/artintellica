import { reactRouter } from "@react-router/dev/vite";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";
import { safeRoutes } from "safe-routes/vite";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  resolve:
    process.env.NODE_ENV === "development"
      ? {}
      : {
          alias: {
            "react-dom/server": "react-dom/server.node",
          },
        },
  plugins: [reactRouter(), safeRoutes(), tsconfigPaths(), tailwindcss()],
});

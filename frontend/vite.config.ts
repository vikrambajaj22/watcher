import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: browser calls /api-proxy/* → FastAPI on8080 (no CORS hassle).
// HMR / React Fast Refresh are on by default (not disabled here).
// If file changes don’t refresh (Docker bind mounts, some network FS), try:
//   VITE_DEV_POLL=1 npm run dev
// Prod: set VITE_API_BASE_URL at build time and configure WATCHER_CORS_ORIGINS on the API.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8501,
    strictPort: true,
    watch: process.env.VITE_DEV_POLL
      ? { usePolling: true, interval: 300 }
      : undefined,
    proxy: {
      "/api-proxy": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api-proxy/, ""),
      },
      // Swagger/ReDoc request /openapi.json from the browser origin. Without this,
      // Vite returns index.html and Swagger shows "does not specify a valid version field".
      "/openapi.json": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
      },
    },
  },
});

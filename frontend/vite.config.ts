import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies /api to the FastAPI backend; in production FastAPI
// serves the built assets, so API calls are same-origin either way.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});

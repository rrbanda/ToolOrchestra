import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    outDir: "../",
    emptyOutDir: false,
    rollupOptions: {
      output: { entryFileNames: "assets/[name].js" },
    },
  },
  server: {
    proxy: {
      "/api": "http://localhost:7860",
    },
  },
});

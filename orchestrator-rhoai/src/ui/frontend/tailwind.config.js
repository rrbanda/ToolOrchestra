/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          0: "#0b0f1a",
          1: "#111827",
          2: "#1a2035",
          3: "#1e293b",
        },
        accent: {
          blue: "#3b82f6",
          orange: "#f97316",
          green: "#22c55e",
          indigo: "#818cf8",
          cyan: "#22d3ee",
          purple: "#c084fc",
          amber: "#fbbf24",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: "#0f172a", card: "#1e293b", input: "#1e293b" },
        border: { DEFAULT: "#334155", bold: "#475569" },
        accent: { DEFAULT: "#3b82f6", hover: "#2563eb" },
        muted: { DEFAULT: "#94a3b8", weak: "#64748b" },
        text: { DEFAULT: "#e2e8f0", bright: "#f1f5f9" },
        danger: "#f87171",
        success: "#22c55e",
      },
    },
  },
  plugins: [],
}

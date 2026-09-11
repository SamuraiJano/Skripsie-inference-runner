import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Keep frontend code on same-origin /api URLs during development. Vite
      // forwards them to the public C++ service; Python remains private.
      '/api': 'http://127.0.0.1:8080',
    },
  },
  build: {
    target: 'es2020',
    cssMinify: 'lightningcss',
    sourcemap: false,
  },
})

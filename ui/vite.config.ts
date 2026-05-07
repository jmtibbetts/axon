import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../web/static/ui',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/socket.io': {
        target: 'http://localhost:7777',
        ws: true,
        changeOrigin: true,
      },
      '/api': {
        target: 'http://localhost:7777',
        changeOrigin: true,
      },
    },
  },
})

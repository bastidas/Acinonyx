/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { FRONTEND_PORT, BACKEND_PORT } from '../configs/appconfig.js'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'node',
  },
  server: {
    port: FRONTEND_PORT,
    proxy: {
      '/api': {
        target: `http://localhost:${BACKEND_PORT}`,
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy, _options) => {
          proxy.on('error', (_err, _req, _res) => {});
        },
      }
    }
  }
})

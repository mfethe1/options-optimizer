import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 4001,
    strictPort: true, // Fail if port is already in use
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
        // No rewrite needed - backend now expects /api prefix
        ws: true, // Enable WebSocket proxying
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log('Proxying:', req.method, req.url, '→', proxyReq.path);
          });
        },
      },
      // WebSocket proxy for agent streaming
      '/ws': {
        target: process.env.VITE_WS_BASE_URL || 'ws://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
  define: {
    // Make API URL available at build time
    'import.meta.env.VITE_API_BASE_URL': JSON.stringify(
      process.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
    ),
  },
})


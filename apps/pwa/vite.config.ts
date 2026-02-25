import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';

export default defineConfig({
  plugins: [preact()],
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8741',
      '/ws': {
        target: 'ws://localhost:8741',
        ws: true,
      },
    },
  },
});

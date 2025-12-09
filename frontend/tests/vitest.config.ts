/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./setup.ts'],
    globals: true,
    // Exclude CSS files and node_modules from test discovery
    exclude: [
      '**/*.css',
      '**/*.scss',
      '**/*.less',
      '**/node_modules/**',
      '**/dist/**',
      '**/cypress/**',
      '**/.{idea,git,cache,output,temp}/**',
      '**/{karma,rollup,webpack,vite,vitest,jest,ava,babel,nyc,cypress,tsup,build}.config.*',
    ],
    // Only include test files from the current directory
    include: [
      '*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}',
      '**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}',
    ],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '../'),
    },
  },
  // Disable CSS processing
  css: {
    postcss: false,
  },
})
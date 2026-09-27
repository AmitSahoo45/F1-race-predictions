const { existsSync } = require('node:fs');
const { chromium } = require('@playwright/test');

const edge = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const chromePath = process.env.CHROME_PATH ?? (process.platform === 'win32' && existsSync(edge) ? edge : chromium.executablePath());
const prefix = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

module.exports = {
  ci: {
    collect: {
      url: [`http://127.0.0.1:3100${prefix}/`],
      startServerCommand: 'node scripts/serve-export.mjs',
      startServerReadyPattern: 'Serving static export',
      chromePath,
      numberOfRuns: 3,
      settings: { chromeFlags: '--headless --no-sandbox' },
    },
    assert: {
      assertions: {
        'largest-contentful-paint': ['error', { maxNumericValue: 2500, aggregationMethod: 'median' }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1, aggregationMethod: 'median' }],
      },
    },
  },
};

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  testIgnore: process.env.APEX_ALLOW_TEST_FIXTURE === '1' ? ['**/journeys.spec.ts'] : ['**/lifecycle.spec.ts'],
  timeout: 45_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  reporter: [['list']],
  snapshotPathTemplate: '{testDir}/__screenshots__/{arg}{ext}',
  use: { ...devices['Desktop Chrome'], channel: process.platform === 'win32' ? 'msedge' : undefined, baseURL: 'http://127.0.0.1:3100', reducedMotion: 'reduce', trace: 'retain-on-failure' },
  webServer: { command: 'node scripts/serve-export.mjs', url: `http://127.0.0.1:3100${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}/`, reuseExistingServer: !process.env.CI, timeout: 30_000 },
});

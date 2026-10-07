import { existsSync } from 'node:fs';
import { expect, test, type Locator } from '@playwright/test';

export async function expectScreenshotIfAvailable(locator: Locator, name: string): Promise<void> {
  await test.step(`Compare screenshot: ${name}`, async (step) => {
    const info = test.info();
    // Playwright defaults to "missing"; only explicit updates should create baselines.
    const updatingSnapshots = ['all', 'changed'].includes(info.config.updateSnapshots);
    step.skip(
      !existsSync(info.snapshotPath(name, { kind: 'screenshot' })) && !updatingSnapshots,
      `Reference screenshot is missing: ${name}`,
    );
    await expect(locator).toHaveScreenshot(name, { animations: 'disabled', maxDiffPixelRatio: 0.08 });
  });
}

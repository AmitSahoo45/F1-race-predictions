import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const prefix = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const visit = (page: Page, path: string) => page.goto(`${prefix}${path}`);
const audit = async (page: Page) => expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);

test('synthetic fixture covers every forecast lifecycle state with accessible content', async ({ page }) => {
  const cases = [
    ['/2026/demonstration/?target=qualifying', 'Demonstration'],
    ['/2026/demonstration/?target=race', 'Not issued'],
    ['/2026/test-scheduled/?target=qualifying', 'Scheduled'],
    ['/2026/test-issued/?target=qualifying', 'Issued'],
    ['/2026/test-awaiting/?target=qualifying', 'Awaiting result'],
    ['/2026/test-reconciled/?target=qualifying', 'Reconciled'],
    ['/2026/test-incomplete/?target=qualifying', 'Reconciled'],
    ['/2026/test-reconstructed/?target=qualifying', 'Reconstructed'],
  ] as const;
  for (const [path, label] of cases) {
    await visit(page, path);
    await expect(page.getByRole('complementary', { name: 'Data status' })).toContainText('SYNTHETIC TEST FIXTURE ONLY');
    await expect(page.locator('.lifecycle strong')).toHaveText(label);
    await audit(page);
  }
});

test('reconciled fixture scores a complete classification and retains retired status', async ({ page }) => {
  await visit(page, '/2026/test-reconciled/?target=qualifying');
  await expect(page.getByRole('heading', { name: 'Event scores' })).toBeVisible();
  await expect(page.getByText('Position MAE')).toBeVisible();
  await expect(page.getByText('P1 Brier')).toBeVisible();
  await page.locator('.tower-row').nth(1).click();
  await expect(page.getByText('retired', { exact: true })).toBeVisible();
  await expect(page.locator('.reconciliation')).toHaveScreenshot('reconciled-event-scores.png', { animations: 'disabled', maxDiffPixelRatio: 0.08 });
  await audit(page);
});

test('incomplete classification exposes DNS and withholds aggregate event scores', async ({ page }) => {
  await visit(page, '/2026/test-incomplete/?target=qualifying');
  await expect(page.getByRole('heading', { name: 'Event scores unavailable' })).toBeVisible();
  await page.locator('.tower-row').first().click();
  await expect(page.getByText('DNS', { exact: true })).toBeVisible();
  await audit(page);
});

test('reconstructed and awaiting displays have stable visual states and keyboard shortcut', async ({ page }) => {
  await visit(page, '/2026/test-reconstructed/?target=qualifying');
  await expect(page.locator('.forecast-heading-line')).toHaveScreenshot('reconstructed-heading.png', { animations: 'disabled', maxDiffPixelRatio: 0.08 });
  await visit(page, '/2026/test-awaiting/?target=qualifying');
  const skip = page.getByRole('link', { name: 'Skip to forecast' });
  await skip.focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#forecast-heading$/);
  await audit(page);
});

test('synthetic public traces switch speed, throttle and brake with an accessible data table', async ({ page }) => {
  await visit(page, '/2026/demonstration/?compare=1');
  await expect(page.getByRole('button', { name: 'Throttle', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Throttle', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Throttle', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('img', { name: /throttle traces/ })).toBeVisible();
  await page.getByRole('button', { name: 'Brake', exact: true }).click();
  await expect(page.getByRole('img', { name: /brake traces/ })).toBeVisible();
  await page.getByText('View telemetry data table').click();
  await expect(page.getByRole('table', { name: 'Selected public speed, throttle and braking samples' })).toBeVisible();
  await audit(page);
});

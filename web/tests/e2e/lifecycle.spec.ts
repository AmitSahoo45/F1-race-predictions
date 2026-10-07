import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import type { SiteData } from '../../src/generated/site-data';
import { expectScreenshotIfAvailable } from './helpers/optional-screenshot';

const prefix = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const visit = (page: Page, path: string) => page.goto(`${prefix}${path}`);
const audit = async (page: Page) => expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);

const lifecycleCases = [
    ['/2026/demonstration/?target=qualifying', 'Demonstration'],
    ['/2026/demonstration/?target=race', 'Not issued'],
    ['/2026/test-scheduled/?target=qualifying', 'Scheduled'],
    ['/2026/test-issued/?target=qualifying', 'Issued'],
    ['/2026/test-awaiting/?target=qualifying', 'Awaiting result'],
    ['/2026/test-reconciled/?target=qualifying', 'Reconciled'],
    ['/2026/test-incomplete/?target=qualifying', 'Reconciled'],
    ['/2026/test-reconstructed/?target=qualifying', 'Reconstructed'],
  ] as const;

for (const [path, label] of lifecycleCases) {
  test(`synthetic ${path} exposes the accessible ${label} lifecycle`, async ({ page }) => {
    await visit(page, path);
    await expect(page.getByRole('complementary', { name: 'Data status' })).toContainText('SYNTHETIC TEST FIXTURE ONLY');
    await expect(page.locator('.lifecycle strong')).toHaveText(label);
    await audit(page);
  });
}

test('client clock advances scheduled to missed cutoff and issued to awaiting result', async ({ page }) => {
  const fixture = JSON.parse(readFileSync(resolve(process.cwd(), 'tests/.generated/lifecycle-site.json'), 'utf8')) as SiteData;
  const scheduled = fixture.events.find((event) => event.id === 'test-scheduled')!;
  const cutoff = Date.parse(scheduled.targets.find((target) => target.target === 'qualifying')!.cutoff_at);
  await page.clock.install({ time: new Date(cutoff - 60_000) });
  await visit(page, '/2026/test-scheduled/?target=qualifying');
  await expect(page.locator('.lifecycle strong')).toHaveText('Scheduled');
  await expect(page.locator('.lifecycle span')).toHaveText('Forecast cutoff in 0h 01m');
  await page.clock.fastForward(61_000);
  await expect(page.locator('.lifecycle strong')).toHaveText('Not issued');

  await visit(page, '/2026/test-issued/?target=qualifying');
  await expect(page.locator('.lifecycle strong')).toHaveText('Issued');
  await page.clock.fastForward(31 * 60_000);
  await expect(page.locator('.lifecycle strong')).toHaveText('Awaiting result');
});

test('reconciled fixture scores a complete classification and retains retired status', async ({ page }) => {
  await visit(page, '/2026/test-reconciled/?target=qualifying');
  await expect(page.getByRole('heading', { name: 'Event scores' })).toBeVisible();
  await expect(page.getByText('Position MAE')).toBeVisible();
  await expect(page.getByText('P1 Brier')).toBeVisible();
  await page.locator('.tower-row').nth(1).click();
  await expect(page.getByText('retired', { exact: true })).toBeVisible();
  await expectScreenshotIfAvailable(page.locator('.reconciliation'), 'reconciled-event-scores.png');
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
  await expectScreenshotIfAvailable(page.locator('.forecast-heading-line'), 'reconstructed-heading.png');
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

test('telemetry loads on demand and changing pairs clears stale traces before a failed request', async ({ page }) => {
  const telemetryRequests: string[] = [];
  page.on('request', (request) => { if (request.url().includes('/telemetry/')) telemetryRequests.push(request.url()); });
  await visit(page, '/2026/demonstration/');
  await expect(page.getByRole('button', { name: 'Open comparison' })).toBeVisible();
  expect(telemetryRequests).toEqual([]);
  await page.getByRole('button', { name: 'Open comparison' }).click();
  await expect(page.getByRole('img', { name: 'speed traces for VEG and CRO across lap distance' })).toBeVisible();

  let failRequest: (() => Promise<void>) | undefined;
  await page.route('**/telemetry/demonstration-driver-03.json', (route) => {
    failRequest = () => route.fulfill({ status: 503, body: 'Unavailable' });
  });
  await page.getByRole('combobox', { name: 'DRIVER A' }).selectOption('LAU');
  await expect.poll(() => Boolean(failRequest)).toBe(true);
  await expect(page.getByRole('status')).toHaveText('Loading public trace summary…');
  await expect(page.locator('.telemetry-chart canvas')).toHaveCount(0);
  await failRequest!();
  await expect(page.getByText('Telemetry traces could not be loaded.')).toBeVisible();
  await audit(page);

  await page.getByRole('combobox', { name: 'DRIVER A' }).selectOption('VEG');
  await expect(page.getByRole('img', { name: 'speed traces for VEG and CRO across lap distance' })).toBeVisible();
});

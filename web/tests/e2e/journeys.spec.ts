import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import type { SiteData } from '../../src/generated/site-data';

const site = JSON.parse(readFileSync(resolve(process.cwd(), '../site-data/site.json'), 'utf8')) as SiteData;
const season = Math.max(...site.events.map((event) => event.season));
const events = site.events.filter((event) => event.season === season).sort((a, b) => a.round - b.round);
const prefix = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const visit = (page: Page, path: string) => page.goto(`${prefix}${path}`);
const audit = async (page: Page) => expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
const errors = new WeakMap<Page, string[]>();

test.beforeEach(({ page }) => {
  const messages: string[] = [];
  errors.set(page, messages);
  page.on('pageerror', (error) => messages.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') messages.push(message.text()); });
});

test.afterEach(({ page }) => {
  expect(errors.get(page), 'No browser runtime or console errors').toEqual([]);
});

test('real-data landing, calendar, weekend and method stay connected', async ({ page }) => {
  await visit(page, '/');
  await expect(page.getByRole('heading', { name: 'Before the lights go out.' })).toBeVisible();
  if (site.demo_notice) await expect(page.getByRole('complementary', { name: 'Data status' })).toContainText(site.demo_notice);
  await page.screenshot({ path: 'artifacts/landing-desktop.png', fullPage: true, animations: 'disabled' });
  await audit(page);

  await page.getByRole('link', { name: 'Calendar', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'The calendar.' })).toBeVisible();
  await expect(page.locator('.calendar-row')).toHaveCount(events.length);
  const seasons = [...new Set(site.events.map((event) => event.season))];
  const seasonNavigation = page.getByRole('navigation', { name: 'Season archive' });
  await expect(seasonNavigation.getByRole('link')).toHaveCount(seasons.length);
  await expect(seasonNavigation.getByRole('link', { name: String(season), exact: true })).toHaveAttribute('aria-current', 'page');
  const oldest = Math.min(...seasons);
  if (oldest !== season) {
    await seasonNavigation.getByRole('link', { name: String(oldest), exact: true }).click();
    await expect(page.locator('.calendar-row')).toHaveCount(site.events.filter((event) => event.season === oldest).length);
    await seasonNavigation.getByRole('link', { name: String(season), exact: true }).click();
    await expect(page.locator('.calendar-row')).toHaveCount(events.length);
  }
  await audit(page);
  await page.locator('.calendar-row').first().click();
  await expect(page.locator('.weekend-hero h1')).toContainText(events[0].name);
  await page.locator('.weekend-tabs').getByRole('button', { name: /Race/ }).click();
  await expect(page).toHaveURL(/target=race/);
  await expect(page.getByRole('heading', { name: 'Race forecast.' })).toBeVisible();
  await page.getByRole('button', { name: 'Open comparison' }).click();
  await expect(page.getByRole('button', { name: 'Close comparison' })).toBeVisible();
  if (events[0].entrants.length >= 3) {
    await page.getByRole('combobox', { name: 'DRIVER A' }).selectOption({ index: 2 });
    await expect(page).toHaveURL(/a=/);
    await expect(page.getByRole('button', { name: 'Close comparison' })).toBeVisible();
  }
  await page.screenshot({ path: 'artifacts/weekend-desktop.png', fullPage: true, animations: 'disabled' });
  await audit(page);
  await page.getByRole('link', { name: 'Methodology', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'What goes into the forecast.' })).toBeVisible();
  await expect(page).toHaveTitle('Methodology & Sources | APEX FORECAST');
  await expect(page.getByRole('link', { name: 'TracingInsights ↗', exact: true })).toBeVisible();
  await audit(page);
});

const examples = new Map<string, { event: SiteData['events'][number]; target: SiteData['events'][number]['targets'][number] }>();
for (const event of site.events) for (const target of event.targets) {
  const key = `${target.state}-${target.target}-${site.forecasts.find((item) => item.id === target.forecast_id)?.kind ?? 'empty'}`;
  if (!examples.has(key)) examples.set(key, { event, target });
}

for (const [state, { event, target }] of examples) {
  test(`real published ${state} target has accessible content`, async ({ page }) => {
    await visit(page, `/${event.season}/${event.id}/?target=${target.target}`);
    await expect(page.getByRole('heading', { name: `${target.target === 'race' ? 'Race' : 'Qualifying'} forecast.` })).toBeVisible();
    const forecast = site.forecasts.find((item) => item.id === target.forecast_id);
    if (forecast) await expect(page.locator('.forecast-tower .tower-item')).toHaveCount(forecast.drivers.length);
    else await expect(page.locator('.forecast-empty')).toBeVisible();
    await audit(page);
  });
}

test('performance exposes the current evaluation and complete static report downloads', async ({ page }) => {
  await visit(page, '/performance/');
  if (site.evaluation.status === 'pending') await expect(page.getByRole('heading', { name: 'Results are pending.' })).toBeVisible();
  else {
    await expect(page.getByRole('table', { name: 'Model performance by season and target' })).toBeVisible();
    for (const target of new Set(site.evaluation.seasons.map((item) => item.target))) {
      const path = `${prefix}/evaluation-reports/${target}.json`;
      await expect(page.getByRole('link', { name: `${target} evaluation report JSON ↗` })).toHaveAttribute('href', path);
      const report = await page.request.get(path);
      expect(report.ok()).toBe(true);
      const expected = JSON.parse(readFileSync(resolve(process.cwd(), `../site-data/evaluation-reports/${target}.json`), 'utf8'));
      expect(await report.json()).toEqual(expected);
    }
  }
  await page.screenshot({ path: 'artifacts/performance-desktop.png', fullPage: true, animations: 'disabled' });
  await audit(page);
});

test('mobile theme, keyboard access and 200% text remain usable with real data', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await visit(page, '/');
  await expect(page.getByRole('heading', { name: 'Before the lights go out.' })).toBeVisible();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  await page.getByRole('button', { name: 'Switch to light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.screenshot({ path: 'artifacts/landing-mobile.png', fullPage: true, animations: 'disabled' });
  await audit(page);
  for (const path of ['/', `/${season}/`, `/${events[0].season}/${events[0].id}/`, '/performance/', '/methodology/']) {
    await visit(page, path);
    await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, `${path} should fit at 200% text`).toBeLessThanOrEqual(1);
    if (path === '/performance/' || path === '/methodology/') await page.screenshot({ path: `artifacts/${path.replaceAll('/', '')}-mobile-200.png`, fullPage: true, animations: 'disabled' });
  }
});

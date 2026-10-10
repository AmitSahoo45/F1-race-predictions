import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { test, expect, type Locator, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import type { SiteData } from '../../src/generated/site-data';

const site = JSON.parse(readFileSync(resolve(process.cwd(), '../site-data/site.json'), 'utf8')) as SiteData;
const season = Math.max(...site.events.map((event) => event.season));
const events = site.events.filter((event) => event.season === season).sort((a, b) => a.round - b.round);
const prefix = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const visit = (page: Page, path: string) => page.goto(`${prefix}${path}`);
const audit = async (page: Page) => expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
const errors = new WeakMap<Page, string[]>();
type Event = SiteData['events'][number];
const eventOrder = (event: Event) => event.season * 100 + event.round;
const observedOutlineEvents = site.events.filter((event) => site.analyses.some((analysis) =>
  analysis.event_id === event.id && analysis.kind === 'observed' && analysis.circuit_points.length >= 3,
)).sort((a, b) => eventOrder(b) - eventOrder(a));
const outlineSource = (event: Event) => observedOutlineEvents.find((source) =>
  source.circuit === event.circuit && eventOrder(source) <= eventOrder(event),
);
const expectOutline = async (artwork: Locator, event: Event) => {
  const source = outlineSource(event);
  expect(source, `${event.id} has an observed same-circuit outline source`).toBeDefined();
  await expect(artwork).toBeVisible();
  await expect(artwork).toHaveAttribute('data-circuit', event.circuit);
  await expect(artwork).toHaveAttribute('data-source-event', source!.id);
  const svg = artwork.getByRole('img');
  await expect(svg).toBeVisible();
  await expect(svg).toHaveAttribute('aria-label', new RegExp(`${event.circuit}.*${source!.season}`));
  const path = artwork.locator('svg path.circuit-main');
  await expect(path).toBeVisible();
  await expect(path).toHaveAttribute('d', /^M.+ Z$/);
  const geometry = await path.evaluate((element) => {
    const outline = element as SVGPathElement;
    const bounds = outline.getBBox();
    return { width: bounds.width, height: bounds.height, length: outline.getTotalLength() };
  });
  expect(geometry.width).toBeGreaterThan(0);
  expect(geometry.height).toBeGreaterThan(0);
  expect(geometry.length).toBeGreaterThan(0);
};

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

test('landing follows the viewer clock through upcoming, ongoing and past weekends', async ({ page }) => {
  const dated = site.events.filter((event) => event.sessions.length).map((event) => ({
    event,
    start: Math.min(...event.sessions.map((session) => Date.parse(session.start))),
    end: Math.max(...event.sessions.map((session) => Date.parse(session.end))),
  })).sort((a, b) => a.start - b.start);
  const [previous, chosen] = dated.slice(-3, -1);
  const latest = [...dated].sort((a, b) => b.end - a.end)[0];
  for (const [clock, label, event] of [
    [previous.end + 60_000, 'THE NEXT CHAPTER', chosen.event],
    [chosen.start + 60 * 60_000, 'THIS WEEKEND', chosen.event],
    [latest.end + 24 * 60 * 60_000, 'FROM THE ARCHIVE', latest.event],
  ] as const) {
    await page.clock.setFixedTime(new Date(clock));
    await visit(page, '/');
    const feature = page.locator('.event-feature');
    await expect(feature.locator('.section-kicker')).toContainText(label);
    await expect(feature.getByRole('heading', { name: event.name })).toBeVisible();
    await expectOutline(page.locator('.hero-art figure.circuit-art'), event);
  }
});

for (const [viewportName, viewport] of [
  ['desktop', { width: 1440, height: 900 }],
  ['mobile', { width: 390, height: 844 }],
] as const) {
  test(`Singapore landing and weekend show the published outline source on ${viewportName}`, async ({ page }) => {
    const event = site.events.find((item) => item.id === '2026-17')!;
    const source = outlineSource(event)!;
    const sourceLabel = `${source.season} ${source.id === event.id ? 'POSITION DATA' : 'REFERENCE OUTLINE'}`;
    expect(event.circuit).toBe('marina_bay');
    await page.setViewportSize(viewport);
    await page.clock.setFixedTime(new Date(Math.min(...event.sessions.map((session) => Date.parse(session.start))) - 60_000));
    await visit(page, '/');
    await expect(page.locator('.event-feature .section-kicker')).toContainText('THE NEXT CHAPTER');
    await expect(page.locator('.event-feature h2')).toHaveText(event.name);
    await expectOutline(page.locator('.hero-art figure.circuit-art'), event);
    await expect(page.locator('.hero-art figcaption')).toBeVisible();
    await expect(page.locator('.hero-art figcaption')).toContainText(sourceLabel);
    await expect(page.locator('.circuit-unavailable')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `artifacts/singapore-landing-${viewportName}.png`, fullPage: true, animations: 'disabled' });

    await page.getByRole('link', { name: 'Explore the weekend' }).click();
    await expect(page).toHaveURL((url) => url.pathname.replace(/\/$/, '') === `${prefix}/${event.season}/${event.id}`);
    await expect(page.locator('.weekend-hero h1')).toContainText(event.name);
    const artwork = page.locator('.weekend-hero figure.circuit-art');
    await expectOutline(artwork, event);
    await expect(artwork.locator('figcaption')).toBeVisible();
    await expect(artwork.locator('figcaption')).toContainText(sourceLabel);
    await page.locator('.weekend-tabs').getByRole('button', { name: /Race/ }).click();
    await expect(page).toHaveURL(/target=race/);
    await expect(page.getByRole('heading', { name: 'Race forecast.' })).toBeVisible();
    await expectOutline(artwork, event);
    await expect(page.locator('.circuit-unavailable')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `artifacts/singapore-weekend-${viewportName}.png`, fullPage: true, animations: 'disabled' });
    await audit(page);
  });
}

test('every published circuit renders a same-circuit outline on its latest weekend', async ({ page }) => {
  test.setTimeout(90_000);
  const latestByCircuit = new Map([...site.events].sort((a, b) => eventOrder(a) - eventOrder(b)).map((event) => [event.circuit, event]));
  for (const [circuit, event] of latestByCircuit) {
    await test.step(`${circuit}: ${event.id}`, async () => {
      await visit(page, `/${event.season}/${event.id}/`);
      await expect(page.locator('.weekend-hero h1')).toContainText(event.name);
      await expectOutline(page.locator('.weekend-hero figure.circuit-art'), event);
      await expect(page.locator('.circuit-unavailable')).toHaveCount(0);
    });
  }
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

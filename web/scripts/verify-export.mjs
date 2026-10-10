import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const out = fileURLToPath(new URL('../out/', import.meta.url));
if (process.env.APEX_SITE_DATA_PATH && process.env.APEX_ALLOW_TEST_FIXTURE !== '1') throw new Error('Test fixture override requires APEX_ALLOW_TEST_FIXTURE=1');
const source = process.env.APEX_SITE_DATA_PATH ? fileURLToPath(new URL(`../${process.env.APEX_SITE_DATA_PATH}`, import.meta.url)) : fileURLToPath(new URL('../../site-data/site.json', import.meta.url));
const site = JSON.parse(await readFile(source, 'utf8'));
const prefix = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
assert(prefix === '' || /^\/[a-zA-Z0-9_-]+$/.test(prefix), 'NEXT_PUBLIC_BASE_PATH must be a single path segment');
const landing = await readFile(join(out, 'index.html'), 'utf8');
assert(landing.includes(`/_next/`) && landing.includes(`${prefix}/_next/`), 'Landing assets lack the configured basePath');
assert(landing.includes(`href="${prefix}/${Math.max(...site.events.map((event) => event.season))}/"`), 'Calendar link lacks the configured basePath');
for (const event of site.events) {
  const path = join(out, String(event.season), event.id, 'index.html');
  const html = await readFile(path, 'utf8');
  assert(html.includes(event.name), `Missing event content in ${path}`);
  const outline = JSON.parse(await readFile(join(out, 'circuits', event.id, 'outline.json'), 'utf8'));
  if (process.env.APEX_ALLOW_TEST_FIXTURE !== '1') {
    assert(outline?.circuit === event.circuit && /^M.+ Z$/.test(outline.path), `Missing or mismatched circuit outline for ${event.id}`);
  }
  assert(html.includes(`${prefix}/_next/`), `Event assets lack basePath in ${path}`);
  assert(html.includes(`${prefix}/og/${event.season}-${event.id}.png`), `OG URL lacks basePath for ${event.id}`);
  await stat(join(out, 'og', `${event.season}-${event.id}.png`));
}
for (const path of ['performance/index.html', 'methodology/index.html']) {
  await stat(join(out, path));
}
if (site.evaluation.status === 'evaluated') {
  for (const target of new Set(site.evaluation.seasons.map((season) => season.target))) {
    await stat(join(out, 'evaluation-reports', `${target}.json`));
  }
}
console.log(`Verified ${site.events.length} exported event routes, Open Graph images, and asset/link basePath ${prefix || '/'} `);

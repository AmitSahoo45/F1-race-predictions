import { readFile, mkdir, unlink, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve, sep } from 'node:path';
import Ajv2020 from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';
import sharp from 'sharp';

if (process.env.APEX_SITE_DATA_PATH && process.env.APEX_ALLOW_TEST_FIXTURE !== '1') throw new Error('Test fixture override requires APEX_ALLOW_TEST_FIXTURE=1');
const source = process.env.APEX_SITE_DATA_PATH ? fileURLToPath(new URL(`../${process.env.APEX_SITE_DATA_PATH}`, import.meta.url)) : fileURLToPath(new URL('../../site-data/site.json', import.meta.url));
const publicDir = resolve(fileURLToPath(new URL('../public/', import.meta.url)));
const manifest = fileURLToPath(new URL('../.apex-generated-assets.json', import.meta.url));
const site = JSON.parse(await readFile(source, 'utf8'));
await mkdir(join(publicDir, 'og'), { recursive: true });
await mkdir(join(publicDir, 'telemetry'), { recursive: true });
await mkdir(join(publicDir, 'evaluation-reports'), { recursive: true });
let previous = [];
try { previous = JSON.parse(await readFile(manifest, 'utf8')); } catch { /* First build has no manifest. */ }
for (const relative of previous) {
  if (!/^((og\/[a-zA-Z0-9_-]+\.(svg|png))|(telemetry\/[a-zA-Z0-9_-]+\.json)|(evaluation-reports\/(qualifying|race)\.json))$/.test(relative)) throw new Error(`Unsafe generated asset path: ${relative}`);
  const path = resolve(publicDir, relative);
  if (!path.startsWith(`${publicDir}${sep}`)) throw new Error(`Generated asset escaped public/: ${relative}`);
  try { await unlink(path); } catch (error) { if (error.code !== 'ENOENT') throw error; }
}
const generated = [];
const escape = (value) => String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
for (const event of site.events) {
  const label = escape(event.name);
  const subline = escape(`${event.season} / ${event.circuit}`);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630"><rect width="1200" height="630" fill="#10120f"/><path d="M60 130H1140M60 520H1140" stroke="#3a3f36"/><text x="60" y="82" fill="#d9ff56" font-family="Arial,sans-serif" font-size="30" font-weight="700" letter-spacing="5">APEX FORECAST</text><text x="60" y="285" fill="#f3f5ef" font-family="Arial,sans-serif" font-size="68" font-weight="700">${label}</text><text x="60" y="365" fill="#aeb5a6" font-family="Arial,sans-serif" font-size="32">${subline}</text><text x="60" y="580" fill="#d9ff56" font-family="Arial,sans-serif" font-size="22" letter-spacing="3">QUALIFYING / RACE FORECAST</text></svg>`;
  await writeFile(join(publicDir, 'og', `${event.season}-${event.id}.png`), await sharp(Buffer.from(svg)).png().toBuffer());
  generated.push(`og/${event.season}-${event.id}.png`);
}
// Traces live beside site.json, one validated file per event; split them per driver for on-demand loading.
const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv);
const validateTelemetry = ajv.compile(JSON.parse(await readFile(fileURLToPath(new URL('../../schemas/telemetry.schema.json', import.meta.url)), 'utf8')));
for (const analysis of site.analyses) {
  let traces = new Map();
  try {
    const telemetry = JSON.parse(await readFile(join(dirname(source), 'telemetry', `${analysis.event_id}.json`), 'utf8'));
    if (!validateTelemetry(telemetry)) throw new Error(`Invalid telemetry for ${analysis.event_id}: ${ajv.errorsText(validateTelemetry.errors)}`);
    if (telemetry.event_id !== analysis.event_id) throw new Error(`Telemetry file names another event: ${analysis.event_id}`);
    traces = new Map(telemetry.traces.map((trace) => [trace.driver_id, trace.points]));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  for (const driver of analysis.drivers) {
    await writeFile(join(publicDir, 'telemetry', `${analysis.event_id}-${driver.driver_id}.json`), JSON.stringify(traces.get(driver.driver_id) ?? []));
    generated.push(`telemetry/${analysis.event_id}-${driver.driver_id}.json`);
  }
}
if (site.evaluation.status === 'evaluated') {
  for (const target of new Set(site.evaluation.seasons.map((season) => season.target))) {
    const report = await readFile(fileURLToPath(new URL(`../../site-data/evaluation-reports/${target}.json`, import.meta.url)), 'utf8');
    JSON.parse(report);
    await writeFile(join(publicDir, 'evaluation-reports', `${target}.json`), report);
    generated.push(`evaluation-reports/${target}.json`);
  }
}
await writeFile(manifest, JSON.stringify(generated, null, 2));
console.log(`Generated ${site.events.length} event previews and on-demand telemetry files`);

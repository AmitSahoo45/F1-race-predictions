import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const source = fileURLToPath(new URL('../tests/fixtures/lifecycle-seed.json', import.meta.url));
const output = fileURLToPath(new URL('../tests/.generated/lifecycle-site.json', import.meta.url));
const site = JSON.parse(await readFile(source, 'utf8'));
const original = site.events.find((event) => event.id === 'demonstration');
const forecast = site.forecasts.find((item) => item.event_id === original?.id && item.target === 'qualifying');
const analysis = site.analyses.find((item) => item.event_id === original?.id);
if (!original || !forecast || !analysis) throw new Error('Demonstration seed event, qualifying forecast and analysis are required');
site.demo_notice = 'SYNTHETIC TEST FIXTURE ONLY. These lifecycle examples are not real issued forecasts, historical backtests, or model evaluations.';
for (const [index, driver] of analysis.drivers.slice(0, 2).entries()) {
  driver.telemetry = Array.from({ length: 48 }, (_, point) => ({
    distance_m: point * 100,
    speed_kph: 170 + index * 7 + Math.round(48 * Math.sin(point / 7)),
    throttle_pct: Math.max(0, Math.min(100, Math.round(60 + 40 * Math.sin(point / 6)))),
    brake: point % 12 < 3,
  }));
}

for (const [slug, state, kind] of [
  ['test-awaiting', 'issued', 'issued'],
  ['test-reconciled', 'issued', 'issued'],
  ['test-incomplete', 'issued', 'issued'],
  ['test-reconstructed', 'reconstructed', 'reconstructed'],
]) {
  const event = structuredClone(original);
  event.id = slug;
  event.name = `Synthetic ${slug.replace('test-', '')} lifecycle`;
  event.round = site.events.length + 1;
  event.sessions = event.sessions.map((session) => ({ ...session, id: session.id.replace(original.id, slug) }));
  event.targets = event.targets.map((target) => target.target === 'qualifying'
    ? { ...target, state, session_id: target.session_id.replace(original.id, slug), forecast_id: `${slug}-qualifying` }
    : { ...target, state: 'unavailable', reason: 'Synthetic test fixture covers qualifying only.', session_id: target.session_id.replace(original.id, slug), forecast_id: null });
  const artifact = structuredClone(forecast);
  artifact.id = `${slug}-qualifying`;
  artifact.event_id = slug;
  artifact.kind = kind;
  artifact.model_version = 'synthetic-lifecycle-fixture';
  artifact.coverage.summary = 'Synthetic lifecycle fixture; no forecast was actually issued.';
  const eventAnalysis = structuredClone(analysis);
  eventAnalysis.event_id = slug;
  eventAnalysis.source = 'Synthetic test fixture data. Circuit outline remains sourced from public FastF1 position data.';
  eventAnalysis.actuals = slug === 'test-reconciled' || slug === 'test-incomplete'
    ? artifact.drivers.map((driver, index) => ({ target: 'qualifying', driver_id: driver.driver_id, position: slug === 'test-incomplete' && index === 0 ? null : driver.rank, status: slug === 'test-incomplete' && index === 0 ? 'DNS' : index === 1 ? 'retired' : 'classified' }))
    : [];
  site.events.push(event);
  site.forecasts.push(artifact);
  site.analyses.push(eventAnalysis);
}

const scheduled = structuredClone(original);
scheduled.id = 'test-scheduled';
scheduled.name = 'Synthetic scheduled lifecycle';
scheduled.round = site.events.length + 1;
const offset = Date.now() + 30 * 24 * 60 * 60 * 1000 - Date.parse(original.sessions[0].start);
scheduled.sessions = scheduled.sessions.map((session) => ({ ...session, id: session.id.replace(original.id, scheduled.id), start: new Date(Date.parse(session.start) + offset).toISOString(), end: new Date(Date.parse(session.end) + offset).toISOString() }));
scheduled.targets = scheduled.targets.map((target) => target.target === 'qualifying'
  ? { ...target, state: 'scheduled', session_id: target.session_id.replace(original.id, scheduled.id), cutoff_at: new Date(Date.parse(scheduled.sessions.find((session) => session.kind === 'Q').start) - 30 * 60 * 1000).toISOString(), forecast_id: null, reason: null }
  : { ...target, state: 'unavailable', session_id: target.session_id.replace(original.id, scheduled.id), forecast_id: null, reason: 'Synthetic test fixture covers qualifying only.' });
site.events.push(scheduled);

const issued = structuredClone(scheduled);
issued.id = 'test-issued';
issued.name = 'Synthetic issued before session lifecycle';
issued.round = site.events.length + 1;
issued.sessions = issued.sessions.map((session) => ({ ...session, id: session.id.replace(scheduled.id, issued.id) }));
issued.targets = issued.targets.map((target) => target.target === 'qualifying'
  ? { ...target, state: 'issued', session_id: target.session_id.replace(scheduled.id, issued.id), forecast_id: 'test-issued-qualifying' }
  : { ...target, session_id: target.session_id.replace(scheduled.id, issued.id) });
const issuedForecast = structuredClone(forecast);
issuedForecast.id = 'test-issued-qualifying';
issuedForecast.event_id = issued.id;
issuedForecast.kind = 'issued';
issuedForecast.cutoff_at = issued.targets.find((target) => target.target === 'qualifying').cutoff_at;
issuedForecast.generated_at = issuedForecast.cutoff_at;
issuedForecast.model_version = 'synthetic-lifecycle-fixture';
issuedForecast.coverage.summary = 'Synthetic lifecycle fixture; no forecast was actually issued.';
site.events.push(issued);
site.forecasts.push(issuedForecast);

await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(site));
console.log('Generated isolated lifecycle fixture at tests/.generated/lifecycle-site.json');

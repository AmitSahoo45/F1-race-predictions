import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { Analysis, Event, SiteData } from '../generated/site-data';
import { selectCircuitOutline } from './circuit-outline';

const points: Analysis['circuit_points'] = [[0, 0], [100, 0], [100, 50], [0, 50]];
const weekend = (id: string, season: number, circuit = 'marina_bay', round = 17) => ({ id, season, round, circuit, name: 'Singapore Grand Prix' }) as Event;
const analysis = (event_id: string, circuit_points = points, kind: Analysis['kind'] = 'observed') => ({ event_id, circuit_points, kind }) as Analysis;
const old = weekend('2024-18', 2024);
const previous = weekend('2025-18', 2025);
const current = weekend('2026-17', 2026);

describe('circuit outline selection', () => {
  it('shows the latest earlier same-circuit outline before a weekend has analysis', () => {
    const result = selectCircuitOutline(current, [previous, current, old], [analysis(old.id), analysis(previous.id)]);
    expect(result).toMatchObject({ sourceEventId: previous.id, sourceSeason: 2025, historical: true, circuit: 'marina_bay' });
    expect(result?.path).toMatch(/^M.+ Z$/);
  });

  it('prefers the selected event geometry and labels it as its own source', () => {
    expect(selectCircuitOutline(current, [old, current], [analysis(old.id), analysis(current.id)]))
      .toMatchObject({ sourceEventId: current.id, sourceSeason: 2026, historical: false });
  });

  it('matches circuit identity rather than a shared Grand Prix name and excludes later events', () => {
    const otherCircuit = weekend('2025-09', 2025, 'catalunya');
    const future = weekend('2027-17', 2027);
    expect(selectCircuitOutline(current, [current, otherCircuit, future], [analysis(otherCircuit.id), analysis(future.id)])).toBeUndefined();
  });

  it('rejects degenerate and nonfinite geometry and skips synthetic fallback candidates', () => {
    const broken: Analysis['circuit_points'] = [[0, 0], [0, 0], [0, 0], [NaN, 2], ['bad', 4]];
    expect(selectCircuitOutline(current, [old, previous, current], [analysis(current.id, broken), analysis(previous.id, points, 'demonstration'), analysis(old.id)]))
      .toMatchObject({ sourceEventId: old.id, sourceSeason: 2024, historical: true });
  });

  it('does not use a later round from the same season for a historical event', () => {
    const later = weekend('2026-18', 2026, 'marina_bay', 18);
    expect(selectCircuitOutline(current, [current, later], [analysis(later.id)])).toBeUndefined();
  });

  it('covers every published track and event without modifying the published bundle', () => {
    const site = JSON.parse(readFileSync(resolve(process.cwd(), '../site-data/site.json'), 'utf8')) as SiteData;
    const before = JSON.stringify(site);
    for (const event of site.events) {
      const outline = selectCircuitOutline(event, site.events, site.analyses);
      expect(outline, `${event.id} (${event.circuit}) needs an outline`).toBeDefined();
      const source = site.events.find((item) => item.id === outline?.sourceEventId)!;
      expect(source.circuit).toBe(event.circuit);
      expect(source.season * 100 + source.round).toBeLessThanOrEqual(event.season * 100 + event.round);
    }
    expect(JSON.stringify(site)).toBe(before);
  });
});

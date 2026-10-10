import type { Analysis, Event } from '../generated/site-data';

export type CircuitOutline = {
  circuit: string;
  path: string;
  sourceEventId: string;
  sourceSeason: number;
  sourceName: string;
  sourceKind: Analysis['kind'];
  historical: boolean;
};

function pathFromPoints(raw: Analysis['circuit_points']): string | undefined {
  const points = raw.filter((point): point is [number, number] => point.length === 2 && point.every((value) => typeof value === 'number' && Number.isFinite(value)));
  if (new Set(points.map(([x, y]) => `${x},${y}`)).size < 3) return undefined;
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  const minX = Math.min(...xs), minY = Math.min(...ys);
  const spanX = Math.max(...xs) - minX, spanY = Math.max(...ys) - minY;
  if (spanX <= 0 || spanY <= 0) return undefined;
  const scale = Math.min(720 / spanX, 490 / spanY);
  const dx = (800 - spanX * scale) / 2, dy = (570 - spanY * scale) / 2;
  return points.map(([x, y], index) => `${index === 0 ? 'M' : 'L'}${(dx + (x - minX) * scale).toFixed(1)},${(dy + (y - minY) * scale).toFixed(1)}`).join(' ') + ' Z';
}

export function selectCircuitOutline(event: Event, events: Event[], analyses: Analysis[]): CircuitOutline | undefined {
  // Reuse artwork only. Historical pace, results and forecasts remain event-specific.
  const earlier = events.filter((item) => item.circuit === event.circuit &&
    (item.season < event.season || (item.season === event.season && item.round < event.round)))
    .sort((a, b) => b.season - a.season || b.round - a.round);
  for (const source of [event, ...earlier]) {
    const historical = source.id !== event.id;
    const analysis = analyses.find((item) => item.event_id === source.id && (!historical || item.kind === 'observed'));
    const path = analysis && pathFromPoints(analysis.circuit_points);
    if (analysis && path) return {
      circuit: source.circuit, path, sourceEventId: source.id, sourceSeason: source.season,
      sourceName: source.name, sourceKind: analysis.kind, historical,
    };
  }
  return undefined;
}

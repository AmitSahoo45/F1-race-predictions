import type { Analysis } from '@/generated/site-data';

function pathFromPoints(raw: Analysis['circuit_points']): string | null {
  const points = raw.filter((point): point is [number, number] => point.length === 2 && point.every((value) => typeof value === 'number' && Number.isFinite(value))) as [number, number][];
  if (points.length < 3) return null;
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs), minY = Math.min(...ys);
  const spanX = Math.max(...xs) - minX || 1, spanY = Math.max(...ys) - minY || 1;
  const scale = Math.min(720 / spanX, 490 / spanY);
  const dx = (800 - spanX * scale) / 2, dy = (570 - spanY * scale) / 2;
  return points.map(([x, y], index) => `${index === 0 ? 'M' : 'L'}${(dx + (x - minX) * scale).toFixed(1)},${(dy + (y - minY) * scale).toFixed(1)}`).join(' ') + ' Z';
}

export function CircuitArtwork({ analysis, circuit, variant = 'hero' }: { analysis?: Analysis; circuit: string; variant?: 'hero' | 'detail' }) {
  const path = analysis ? pathFromPoints(analysis.circuit_points) : null;
  if (!path) return <div className={`circuit-unavailable circuit-${variant}`} role="img" aria-label={`Circuit geometry for ${circuit} is not yet available`}>
    <span className="circuit-cross" aria-hidden="true">＋</span><p>{circuit}</p><small>Verified circuit geometry pending</small>
  </div>;
  return <figure className={`circuit-art circuit-${variant}`}>
    <svg viewBox="0 0 800 570" role="img" aria-label={`${circuit} circuit outline derived from public position data`} preserveAspectRatio="xMidYMid meet">
      <path className="circuit-shadow" d={path} fill="none" strokeWidth="26" strokeLinejoin="round" />
      <path className="circuit-main" d={path} fill="none" strokeWidth="3" strokeLinejoin="round" />
    </svg>
    <figcaption>TRACK GEOMETRY / {circuit.toUpperCase()} <span>POSITION DATA</span></figcaption>
  </figure>;
}

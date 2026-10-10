import type { CircuitOutline } from '@/lib/circuit-outline';

export function CircuitArtwork({ outline, circuit, variant = 'hero', loading = false }: { outline?: CircuitOutline; circuit: string; variant?: 'hero' | 'detail'; loading?: boolean }) {
  if (!outline || outline.circuit !== circuit) return <div className={`circuit-unavailable circuit-${variant}`} role="img" aria-label={loading ? `Loading ${circuit} circuit outline` : `Circuit geometry for ${circuit} is not yet available`}>
    <span className="circuit-cross" aria-hidden="true">＋</span><p>{circuit}</p><small>{loading ? 'Loading circuit outline' : 'Verified circuit geometry pending'}</small>
  </div>;
  const source = `${outline.sourceSeason} ${outline.sourceName}`;
  const label = outline.sourceKind === 'demonstration' ? 'DEMONSTRATION' : outline.historical ? 'REFERENCE OUTLINE' : 'POSITION DATA';
  return <figure className={`circuit-art circuit-${variant}`} data-circuit={outline.circuit} data-source-event={outline.sourceEventId}>
    <svg viewBox="0 0 800 570" role="img" aria-label={`${circuit} circuit outline from ${source}${outline.historical ? ' (historical reference)' : ''}${outline.sourceKind === 'demonstration' ? ' (demonstration)' : ''}`} preserveAspectRatio="xMidYMid meet">
      <path className="circuit-shadow" d={outline.path} fill="none" strokeWidth="26" strokeLinejoin="round" />
      <path className="circuit-main" d={outline.path} fill="none" strokeWidth="3" strokeLinejoin="round" />
    </svg>
    <figcaption>TRACK GEOMETRY / {circuit.toUpperCase()} <span>{outline.sourceSeason} {label}</span></figcaption>
  </figure>;
}

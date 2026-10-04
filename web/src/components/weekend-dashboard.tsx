'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { MotionConfig, motion } from 'motion/react';
import type { ActualResult, Analysis, DriverAnalysis, DriverPrediction, Event, Forecast, TargetState } from '@/generated/site-data';
import { CircuitArtwork } from './circuit-artwork';
import { TimeLabel } from './time-label';
import { calculateEventScores, deriveForecastState, formatProbability, readComparison, writeComparison } from '@/lib/forecast';

const TelemetryCanvas = dynamic(() => import('./telemetry-canvas'), { ssr: false });
type LeanAnalysis = Omit<Analysis, 'drivers'> & { drivers: Omit<DriverAnalysis, 'telemetry'>[] };
type Target = 'qualifying' | 'race';

function PositionStrip({ probabilities, actual }: { probabilities: number[]; actual?: number | null }) {
  const max = Math.max(...probabilities, 0.01);
  return <div className="distribution-wrap"><div className="distribution" role="img" aria-label={`Position probability distribution${actual ? `; actual position ${actual}` : ''}`}>
    {probabilities.map((probability, index) => <span key={index} className={actual === index + 1 ? 'actual-position' : ''} style={{ height: `${Math.max(7, probability / max * 100)}%` }} title={`P${index + 1}: ${formatProbability(probability)}`} />)}
  </div><span className="strip-label">P1 <i>P{probabilities.length}</i></span></div>;
}

function ForecastRow({ prediction, event, target, actual }: { prediction: DriverPrediction; event: Event; target: Target; actual?: ActualResult }) {
  const [open, setOpen] = useState(false);
  const driver = event.entrants.find((item) => item.id === prediction.driver_id);
  if (!driver) return null;
  return <motion.li layout className={`tower-item ${open ? 'is-open' : ''}`}>
    <button className="tower-row" type="button" aria-expanded={open} onClick={() => setOpen(!open)}>
      <span className="tower-rank">{String(prediction.rank).padStart(2, '0')}</span>
      <span className="tower-driver"><span className="team-line" style={{ background: driver.color }} aria-hidden="true" /><span><strong>{driver.code}</strong><small>{driver.name} / {driver.team}</small></span></span>
      <span className="tower-range">P{prediction.p10}–P{prediction.p90}<small>80% range</small></span>
      <span className="tower-prob">{formatProbability(target === 'qualifying' ? prediction.win_probability : prediction.podium_probability)}<small>{target === 'qualifying' ? 'Qualifying P1' : 'Podium'}</small></span>
      <PositionStrip probabilities={prediction.position_probabilities} actual={actual?.position} />
      <span className="tower-chevron" aria-hidden="true">{open ? '−' : '+'}</span>
    </button>
    {open && <motion.div className="tower-detail" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      <div><span className="detail-label">EXPECTED POSITION</span><strong>P{prediction.expected_position.toFixed(1)}</strong></div>
      <div><span className="detail-label">{target === 'qualifying' ? 'QUALIFYING P1' : 'WIN'}</span><strong>{formatProbability(prediction.win_probability)}</strong></div>
      <div><span className="detail-label">PODIUM</span><strong>{formatProbability(prediction.podium_probability)}</strong></div>
      <div><span className="detail-label">TOP TEN <abbr title="A finishing-position proxy, not the exact probability of championship points">PROXY</abbr></span><strong>{formatProbability(prediction.top10_probability)}</strong></div>
      {actual && <div><span className="detail-label">OBSERVED RESULT</span><strong>{actual.position == null ? actual.status : `P${actual.position}`}</strong><small>{actual.position == null ? 'No classified position' : actual.status}</small></div>}
      {prediction.explanations.length > 0 && <div className="explanations"><span className="detail-label">MODEL SCORE CONTRIBUTORS</span><ul>{prediction.explanations.map((explanation) => <li key={explanation.group}>{explanation.group}: {explanation.contribution >= 0 ? '+' : ''}{explanation.contribution.toFixed(2)}</li>)}</ul><small>These describe the model score, not causes of the result or probability.</small></div>}
    </motion.div>}
  </motion.li>;
}

function ForecastPanel({ event, target, state, forecast, analysis }: { event: Event; target: Target; state: TargetState; forecast?: Forecast; analysis?: LeanAnalysis }) {
  const [now, setNow] = useState<number>();
  useEffect(() => { setNow(Date.now()); const interval = window.setInterval(() => setNow(Date.now()), 60_000); return () => window.clearInterval(interval); }, []);
  const actuals = analysis?.actuals.filter((item) => item.target === target) ?? [];
  const scores = forecast && actuals.length ? calculateEventScores(forecast.drivers, actuals) : null;
  const session = event.sessions.find((item) => item.id === state.session_id);
  const lifecycle = deriveForecastState(state, actuals.length > 0, now ?? Date.parse(state.cutoff_at) - 60_000, forecast?.kind, session?.start);
  return <section className="forecast-section" aria-labelledby="forecast-heading">
    <div className="section-kicker"><span>THE FORECAST / {target.toUpperCase()}</span><span>{session?.kind ?? target.toUpperCase()} SESSION</span></div>
    <div className="forecast-heading-line"><div><h2 id="forecast-heading">{target === 'qualifying' ? 'Qualifying' : 'Race'} forecast<span className="accent-dot">.</span></h2><p>{target === 'qualifying' ? 'Predicted qualifying order and P1 probability. Qualifying P1 is distinct from starting-grid pole.' : 'Predicted finishing order, win and podium probability, and top-ten position proxy.'}</p></div><div className={`lifecycle lifecycle-${lifecycle.tone}`}><strong>{lifecycle.label}</strong><span>{lifecycle.detail}</span></div></div>
    {session && <div className="session-line"><span>SESSION START</span><TimeLabel iso={session.start} trackZone={event.timezone} detail /><span>FORECAST CUTOFF</span><TimeLabel iso={state.cutoff_at} trackZone={event.timezone} detail /></div>}
    {!forecast ? <div className="forecast-empty"><span aria-hidden="true">↗</span><h3>{lifecycle.label}</h3><p>{lifecycle.detail}</p><p>Eligible sessions: {event.sessions.filter((item) => Date.parse(item.end) < Date.parse(state.cutoff_at)).map((item) => item.kind).join(' · ') || 'None yet'}</p></div> : <>
      <div className="forecast-context"><div><span className="detail-label">INPUT COVERAGE</span><p>{forecast.coverage.summary}</p>{forecast.coverage.flags.length > 0 && <small>{forecast.coverage.flags.join(' · ')}</small>}</div><div><span className="detail-label">MODEL</span><p>{forecast.model_kind === 'baseline' ? 'Baseline' : 'CatBoost'} / {forecast.model_version}</p><small>{forecast.sample_count?.toLocaleString() ?? '20,000'} sampled orders · fixed seed</small></div><div><span className="detail-label">FROZEN AT</span><p><TimeLabel iso={forecast.generated_at} /></p><small>Training cutoff <TimeLabel iso={forecast.training_cutoff} /></small></div></div>
      {actuals.length > 0 && <section className="reconciliation" aria-label="Event scores"><div><span className="detail-label">OBSERVED VS FORECAST</span><h3>{scores ? 'Event scores' : 'Event scores unavailable'}</h3></div>{scores ? <dl><div><dt>Position MAE</dt><dd>{scores.positionMae.toFixed(2)}</dd></div><div><dt>{target === 'qualifying' ? 'P1' : 'Win'} Brier</dt><dd>{scores.winBrier.toFixed(3)}</dd></div><div><dt>Podium Brier</dt><dd>{scores.podiumBrier.toFixed(3)}</dd></div><div><dt>Top ten Brier</dt><dd>{scores.top10Brier.toFixed(3)}</dd></div></dl> : <p>Complete, one-to-one classified positions are required. Missing positions, DNS, or duplicate classifications prevent a full-event score; individual statuses remain visible in driver details.</p>}</section>}
      <div className="tower-head" aria-hidden="true"><span>RANK</span><span>DRIVER / TEAM</span><span>POSITION RANGE</span><span>{target === 'qualifying' ? 'P1' : 'PODIUM'}</span><span>POSITION DISTRIBUTION</span></div>
      <ol className="forecast-tower">{forecast.drivers.map((prediction) => <ForecastRow key={prediction.driver_id} prediction={prediction} event={event} target={target} actual={actuals.find((item) => item.driver_id === prediction.driver_id)} />)}</ol>
      <details className="detail-disclosure"><summary>Read the position probabilities as a table</summary><div className="table-scroll"><table><caption>{target} position distribution by driver</caption><thead><tr><th>Driver</th>{forecast.drivers[0]?.position_probabilities.map((_, index) => <th key={index}>P{index + 1}</th>)}</tr></thead><tbody>{forecast.drivers.map((driver) => <tr key={driver.driver_id}><th>{event.entrants.find((item) => item.id === driver.driver_id)?.code ?? driver.driver_id}</th>{driver.position_probabilities.map((value, index) => <td key={index}>{formatProbability(value)}</td>)}</tr>)}</tbody></table></div></details>
      <p className="fine-print">Probabilities are estimates from the same sampled orders. Whole percentages are rounded; &lt;1% and &gt;99% avoid implied certainty. An 80% position range spans the 10th to 90th percentile.</p>
    </>}
  </section>;
}

function Comparison({ event, analysis, query, updateQuery }: { event: Event; analysis?: LeanAnalysis; query: string; updateQuery: (query: string) => void }) {
  const entrants = event.entrants;
  const validPair = readComparison(query, entrants.map((driver) => driver.code));
  const pair: [string, string] = validPair ?? [entrants[0]?.code ?? '', entrants[1]?.code ?? ''];
  const compareParam = new URLSearchParams(query).get('compare');
  const open = compareParam === '1' || (compareParam !== '0' && validPair !== null);
  function toggle() { const params = new URLSearchParams(query); params.set('compare', open ? '0' : '1'); updateQuery(params.toString()); }
  function choose(index: number, code: string) {
    const next: [string, string] = index === 0 ? [code, pair[1]] : [pair[0], code];
    if (next[0] === next[1]) return;
    updateQuery(writeComparison(query, next));
  }
  const a = entrants.find((driver) => driver.code === pair[0]);
  const b = entrants.find((driver) => driver.code === pair[1]);
  const aData = analysis?.drivers.find((driver) => driver.driver_id === a?.id);
  const bData = analysis?.drivers.find((driver) => driver.driver_id === b?.id);
  return <section className="comparison-section" aria-labelledby="comparison-heading"><div className="section-kicker"><span>THE COMPARISON</span><span>02 / ANALYSIS</span></div><div className="comparison-heading"><div><h2 id="comparison-heading">Head to head<span className="accent-dot">.</span></h2><p>Practice pace, long runs and selected public telemetry samples. Pair selections are shareable.</p></div><button type="button" className="outline-button" onClick={toggle} aria-expanded={open}>{open ? 'Close comparison' : 'Open comparison'} <span aria-hidden="true">↗</span></button></div>
    {open && <div className="comparison-content"><div className="comparison-selectors"><label>DRIVER A<select value={pair[0]} onChange={(event) => choose(0, event.target.value)}>{entrants.map((driver) => <option value={driver.code} key={driver.id}>{driver.code} — {driver.name}</option>)}</select></label><span aria-hidden="true">VS</span><label>DRIVER B<select value={pair[1]} onChange={(event) => choose(1, event.target.value)}>{entrants.map((driver) => <option value={driver.code} key={driver.id}>{driver.code} — {driver.name}</option>)}</select></label></div>
      {!aData || !bData ? <p className="empty-state">Comparable practice data is unavailable for this pair.</p> : <><div className="pace-grid"><div><span>PRACTICE PACE</span><strong>{aData.practice_pace_s == null ? '—' : `${aData.practice_pace_s.toFixed(3)} s`}</strong><small>{a?.name}</small></div><div><span>PRACTICE PACE</span><strong>{bData.practice_pace_s == null ? '—' : `${bData.practice_pace_s.toFixed(3)} s`}</strong><small>{b?.name}</small></div><div><span>LONG-RUN PACE</span><strong>{aData.long_run_pace_s == null ? '—' : `${aData.long_run_pace_s.toFixed(3)} s`}</strong><small>{a?.name}</small></div><div><span>LONG-RUN PACE</span><strong>{bData.long_run_pace_s == null ? '—' : `${bData.long_run_pace_s.toFixed(3)} s`}</strong><small>{b?.name}</small></div></div>
      <div className="stint-grid">{[aData, bData].map((driverData, index) => <div key={driverData.driver_id}><h3>{index === 0 ? a?.name : b?.name} / tyre stints</h3>{driverData.stints.length ? <ul>{driverData.stints.map((stint, i) => <li key={i}><span>{stint.compound}</span><span>{stint.laps} laps</span><strong>{stint.pace_s.toFixed(3)} s</strong></li>)}</ul> : <p>No stint summary available.</p>}</div>)}</div>
      {analysis?.kind !== 'demonstration' && <p className="fine-print">Trace comparisons align selected fastest laps by distance. Each driver uses their latest practice session with available telemetry, so the laps may come from different sessions. Tyre stints span the available practice sessions.</p>}
      <TelemetryCanvas eventId={event.id} a={a!.id} b={b!.id} aName={a!.code} bName={b!.code} />
      <p className="fine-print">{analysis?.kind === 'demonstration' ? 'Synthetic telemetry demonstrates the interface only.' : 'Public telemetry is a selected summary, not the complete sensor stream held by teams.'} Source: {analysis?.source}.</p></>}
    </div>}
  </section>;
}

export function WeekendDashboard({ event, forecasts, analysis }: { event: Event; forecasts: Forecast[]; analysis?: LeanAnalysis }) {
  const router = useRouter();
  const pathname = usePathname();
  const search = useSearchParams();
  const query = search.toString();
  const active: Target = search.get('target') === 'race' ? 'race' : 'qualifying';
  const target = event.targets.find((item) => item.target === active);
  const forecast = forecasts.find((item) => item.target === active);
  function updateQuery(next: string) { router.replace(`${pathname}?${next}`, { scroll: false }); }
  function selectTarget(next: Target) { const params = new URLSearchParams(query); params.set('target', next); updateQuery(params.toString()); }
  return <MotionConfig reducedMotion="user"><div className="frame weekend-page"><a className="skip-link" href="#forecast-heading">Skip to forecast</a><div className="weekend-top"><Link href={`/${event.season}/`} className="back-link">← {event.season} calendar</Link><span>ROUND {String(event.round).padStart(2, '0')} / {event.country.toUpperCase()}</span></div>
    <div className="weekend-hero"><div><h1>{event.name}<span className="accent-dot">.</span></h1><p>{event.circuit} <span>/</span> {event.country}</p></div><CircuitArtwork analysis={analysis as Analysis | undefined} circuit={event.circuit} variant="detail" /></div>
    <nav className="weekend-tabs" aria-label="Forecast target"><button type="button" onClick={() => selectTarget('qualifying')} aria-current={active === 'qualifying' ? 'page' : undefined}>01 <strong>Qualifying</strong></button><button type="button" onClick={() => selectTarget('race')} aria-current={active === 'race' ? 'page' : undefined}>02 <strong>Race</strong></button></nav>
    {target && <ForecastPanel key={active} event={event} target={active} state={target} forecast={forecast} analysis={analysis} />}
    <div className="section-kicker session-kicker"><span>WEEKEND CHRONOLOGY</span><span>ALL TIMES IN YOUR TIME ZONE</span></div><ol className="session-list">{event.sessions.map((session, index) => <li key={session.id}><span>{String(index + 1).padStart(2, '0')}</span><strong>{session.kind}</strong><TimeLabel iso={session.start} trackZone={event.timezone} detail /></li>)}</ol>
    <Comparison event={event} analysis={analysis} query={query} updateQuery={updateQuery} />
  </div></MotionConfig>;
}

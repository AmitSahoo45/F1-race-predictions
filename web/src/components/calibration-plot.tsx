import { scaleLinear } from 'd3-scale';
import type { CalibrationBin } from '@/generated/site-data';

const marks = { win: '#d8ff4c', podium: '#ad8aff', top10: '#63c7c0' };

export function CalibrationPlot({ season, target, bins }: { season: number; target: string; bins: CalibrationBin[] }) {
  const x = scaleLinear().domain([0, 1]).range([48, 398]);
  const y = scaleLinear().domain([0, 1]).range([255, 18]);
  return <figure className="calibration-plot"><svg viewBox="0 0 440 300" role="img" aria-labelledby={`cal-title-${season}-${target} cal-desc-${season}-${target}`}>
    <title id={`cal-title-${season}-${target}`}>{`${season} ${target} reliability plot`}</title>
    <desc id={`cal-desc-${season}-${target}`}>Predicted probability is on the horizontal axis and observed frequency on the vertical axis. A diagonal guide marks perfect calibration. Exact values follow in the data table.</desc>
    {[0, .25, .5, .75, 1].map((value) => <g key={value}><line x1={x(value)} x2={x(value)} y1={y(0)} y2={y(1)} stroke="rgb(var(--line))" strokeWidth=".7" /><line x1={x(0)} x2={x(1)} y1={y(value)} y2={y(value)} stroke="rgb(var(--line))" strokeWidth=".7" /><text x={x(value)} y="280" textAnchor="middle" fill="rgb(var(--muted))" fontSize="11">{Math.round(value * 100)}%</text><text x="37" y={y(value) + 4} textAnchor="end" fill="rgb(var(--muted))" fontSize="11">{Math.round(value * 100)}%</text></g>)}
    <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="rgb(var(--ink))" strokeWidth="1.5" strokeDasharray="5 5" />
    {bins.map((bin, index) => <g key={`${bin.outcome}-${index}`}><circle cx={x(bin.predicted)} cy={y(bin.observed)} r={Math.max(5, Math.min(11, Math.sqrt(bin.count)))} fill={marks[bin.outcome]} stroke="rgb(var(--paper))" strokeWidth="2"><title>{`${bin.outcome}: predicted ${Math.round(bin.predicted * 100)}%, observed ${Math.round(bin.observed * 100)}%, ${bin.count} samples`}</title></circle></g>)}
  </svg><figcaption>{season} / {target} <span className="cal-legend"><i style={{ background: marks.win }} /> Win <i style={{ background: marks.podium }} /> Podium <i style={{ background: marks.top10 }} /> Top ten</span></figcaption></figure>;
}

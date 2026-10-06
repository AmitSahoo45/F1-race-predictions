'use client';

import { useEffect, useRef, useState } from 'react';
import type { TelemetryPoint } from '@/generated/telemetry';

export default function TelemetryCanvas({ eventId, a, b, aName, bName }: { eventId: string; a: string; b: string; aName: string; bName: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [traces, setTraces] = useState<[TelemetryPoint[], TelemetryPoint[]] | null>(null);
  const [error, setError] = useState(false);
  const [channel, setChannel] = useState<'speed' | 'throttle' | 'brake'>('speed');
  useEffect(() => {
    const controller = new AbortController();
    const base = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
    Promise.all([a, b].map(async (id) => {
      const response = await fetch(`${base}/telemetry/${eventId}-${id}.json`, { signal: controller.signal });
      if (!response.ok) throw new Error(`Trace unavailable: ${id}`);
      return response.json() as Promise<TelemetryPoint[]>;
    })).then((result) => { setTraces(result as [TelemetryPoint[], TelemetryPoint[]]); setError(false); }).catch(() => { if (!controller.signal.aborted) setError(true); });
    return () => controller.abort();
  }, [eventId, a, b]);
  useEffect(() => {
    if (!traces || !canvas.current) return;
    const element = canvas.current;
    const scale = window.devicePixelRatio || 1;
    const width = element.clientWidth, height = element.clientHeight;
    element.width = Math.round(width * scale); element.height = Math.round(height * scale);
    const ctx = element.getContext('2d');
    if (!ctx) return;
    ctx.scale(scale, scale);
    ctx.clearRect(0, 0, width, height);
    const all = traces.flat();
    if (!all.length) return;
    const minDistance = Math.min(...all.map((p) => p.distance_m)), maxDistance = Math.max(...all.map((p) => p.distance_m));
    const value = (point: TelemetryPoint) => channel === 'speed' ? point.speed_kph : channel === 'throttle' ? point.throttle_pct : Number(point.brake);
    const minValue = channel === 'speed' ? Math.min(...all.map(value)) : 0;
    const maxValue = channel === 'speed' ? Math.max(...all.map(value)) : channel === 'throttle' ? 100 : 1;
    ctx.strokeStyle = '#7b8073'; ctx.lineWidth = 1;
    for (let tick = 0; tick < 4; tick++) { const y = 14 + tick * (height - 30) / 3; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
    traces.forEach((trace, index) => {
      if (!trace.length) return;
      ctx.beginPath(); ctx.strokeStyle = index === 0 ? '#d5ff4f' : '#a98cff'; ctx.lineWidth = 2;
      trace.forEach((point, pointIndex) => {
        const x = 6 + ((point.distance_m - minDistance) / (maxDistance - minDistance || 1)) * (width - 12);
        const y = height - 12 - ((value(point) - minValue) / (maxValue - minValue || 1)) * (height - 28);
        if (pointIndex === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }); ctx.stroke();
    });
  }, [traces, channel]);
  if (error) return <p className="empty-state">Telemetry traces could not be loaded.</p>;
  if (!traces) return <p role="status" className="empty-state">Loading public trace summary…</p>;
  if (!traces[0].length || !traces[1].length) return <p className="empty-state">No telemetry trace is available for this pair.</p>;
  return <div className="telemetry-chart"><div className="telemetry-toolbar"><div className="chart-legend"><span className="legend-a">{aName}</span><span className="legend-b">{bName}</span><span>{channel === 'speed' ? 'Speed (km/h)' : channel === 'throttle' ? 'Throttle (%)' : 'Brake on/off'} / distance</span></div><div className="channel-controls" role="group" aria-label="Telemetry channel">{(['speed', 'throttle', 'brake'] as const).map((option) => <button key={option} type="button" aria-pressed={channel === option} onClick={() => setChannel(option)}>{option[0].toUpperCase() + option.slice(1)}</button>)}</div></div><canvas ref={canvas} role="img" aria-label={`${channel} traces for ${aName} and ${bName} across lap distance`} /><details><summary>View telemetry data table</summary><div className="table-scroll"><table><caption>Selected public speed, throttle and braking samples</caption><thead><tr><th>Driver</th><th>Distance</th><th>Speed</th><th>Throttle</th><th>Brake</th></tr></thead><tbody>{traces.flatMap((trace, i) => trace.filter((_, index) => index % Math.max(1, Math.ceil(trace.length / 24)) === 0).map((point) => <tr key={`${i}-${point.distance_m}`}><th>{i === 0 ? aName : bName}</th><td>{Math.round(point.distance_m)} m</td><td>{Math.round(point.speed_kph)} km/h</td><td>{Math.round(point.throttle_pct)}%</td><td>{point.brake ? 'Yes' : 'No'}</td></tr>))}</tbody></table></div></details></div>;
}

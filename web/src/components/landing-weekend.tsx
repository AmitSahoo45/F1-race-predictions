'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { CircuitArtwork } from '@/components/circuit-artwork';
import type { CircuitOutline } from '@/lib/circuit-outline';
import { TimeLabel } from '@/components/time-label';
import { selectLandingEvent, type LandingEvent } from '@/lib/landing-event';

type Artwork = { eventId: string; outline?: CircuitOutline };

export function LandingWeekend({ events, initialArtwork }: { events: LandingEvent[]; initialArtwork?: Artwork }) {
  const [now, setNow] = useState<number | null>(null);
  const [artwork, setArtwork] = useState(initialArtwork);
  useEffect(() => {
    const refresh = () => setNow(Date.now());
    refresh();
    const timer = window.setInterval(refresh, 30_000);
    window.addEventListener('focus', refresh);
    document.addEventListener('visibilitychange', refresh);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener('focus', refresh);
      document.removeEventListener('visibilitychange', refresh);
    };
  }, []);
  const { event, mode } = selectLandingEvent(events, now);
  useEffect(() => {
    if (!event || artwork?.eventId === event.id) return;
    const controller = new AbortController();
    const load = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}/circuits/${encodeURIComponent(event.id)}/outline.json`, { signal: controller.signal });
        if (!response.ok) throw new Error('Circuit outline unavailable');
        const outline = await response.json() as CircuitOutline | null;
        if (outline && outline.circuit !== event.circuit) throw new Error('Circuit outline does not match the weekend');
        if (!controller.signal.aborted) setArtwork({ eventId: event.id, outline: outline ?? undefined });
      } catch {
        if (!controller.signal.aborted) setArtwork({ eventId: event.id });
      }
    };
    void load();
    return () => controller.abort();
  }, [event, artwork]);
  const loadingOutline = Boolean(event && artwork?.eventId !== event.id);
  const race = event?.sessions.find((session) => session.kind === 'R');
  const href = event ? `/${event.season}/${event.id}/` : '/2026/';
  const label = { upcoming: 'THE NEXT CHAPTER', ongoing: 'THIS WEEKEND', archive: 'FROM THE ARCHIVE', calendar: 'THE CALENDAR' }[mode];
  return <>
    <section className="hero frame" aria-labelledby="hero-title">
      <div className="hero-copy">
        <h1 id="hero-title">Before the<br />lights go out<span className="hero-period">.</span></h1>
        <p className="hero-intro landing-reveal">Qualifying and race forecasts from public data, with the uncertainty and evidence in view.</p>
        <div className="hero-actions landing-reveal"><Link className="action-primary" href={href}>Explore the weekend <span aria-hidden="true">↗</span></Link><Link className="action-text" href="/methodology/">How it works <span aria-hidden="true">→</span></Link></div>
      </div>
      <div className="hero-art landing-reveal" aria-busy={loadingOutline}><CircuitArtwork outline={loadingOutline ? undefined : artwork?.outline} circuit={event?.circuit ?? 'Circuit'} loading={loadingOutline} /></div>
      <div className="hero-bottom landing-reveal"><span>01 / FORECAST THE GRID</span><span>SCROLL TO EXPLORE ↓</span></div>
    </section>
    <section className="event-feature frame scroll-reveal" aria-labelledby="next-event-title">
      <div className="section-kicker"><span>{label}</span><span>01 / 03</span></div>
      <div className="feature-grid"><div><h2 id="next-event-title">{event?.name ?? 'Calendar pending'}</h2><p>{event ? `${event.circuit}, ${event.country}` : 'Schedule pending'}</p></div><div className="feature-meta"><p>{race && event ? <TimeLabel iso={race.start} trackZone={event.timezone} detail /> : 'Schedule pending'}</p><Link href={href}>Open race weekend <span aria-hidden="true">↗</span></Link></div></div>
    </section>
  </>;
}

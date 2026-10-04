'use client';

import Link from 'next/link';
import { useEffect, useState, type ReactNode } from 'react';
import { TimeLabel } from '@/components/time-label';
import { selectLandingEvent, type LandingEvent } from '@/lib/landing-event';

export function LandingWeekend({ events, artwork }: { events: LandingEvent[]; artwork: ReactNode }) {
  const [now, setNow] = useState<number | null>(null);
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
      <div className="hero-art landing-reveal">{artwork}</div>
      <div className="hero-bottom landing-reveal"><span>01 / FORECAST THE GRID</span><span>SCROLL TO EXPLORE ↓</span></div>
    </section>
    <section className="event-feature frame scroll-reveal" aria-labelledby="next-event-title">
      <div className="section-kicker"><span>{label}</span><span>01 / 03</span></div>
      <div className="feature-grid"><div><h2 id="next-event-title">{event?.name ?? 'Calendar pending'}</h2><p>{event ? `${event.circuit}, ${event.country}` : 'Schedule pending'}</p></div><div className="feature-meta"><p>{race && event ? <TimeLabel iso={race.start} trackZone={event.timezone} detail /> : 'Schedule pending'}</p><Link href={href}>Open race weekend <span aria-hidden="true">↗</span></Link></div></div>
    </section>
  </>;
}

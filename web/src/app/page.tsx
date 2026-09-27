import Link from 'next/link';
import { CircuitArtwork } from '@/components/circuit-artwork';
import { LandingMotion } from '@/components/landing-motion';
import { TimeLabel } from '@/components/time-label';
import { getSiteData } from '@/lib/data';

export default function Home() {
  const data = getSiteData();
  const event = data.events.find((item) => item.targets.some((target) => target.state === 'scheduled')) ?? data.events[0];
  const artAnalysis = data.analyses.find((item) => item.circuit_points.length > 2);
  const artEvent = data.events.find((item) => item.id === artAnalysis?.event_id);
  return <main id="main"><LandingMotion />
    <section className="hero frame" aria-labelledby="hero-title">
      <div className="hero-copy">
        <h1 id="hero-title">Before the<br />lights go out<span className="hero-period">.</span></h1>
        <p className="hero-intro landing-reveal">Qualifying and race forecasts from public data, with the uncertainty and evidence in view.</p>
        <div className="hero-actions landing-reveal"><Link className="action-primary" href={event ? `/${event.season}/${event.id}/` : '/2026/'}>Explore the weekend <span aria-hidden="true">↗</span></Link><Link className="action-text" href="/methodology/">How it works <span aria-hidden="true">→</span></Link></div>
      </div>
      <div className="hero-art landing-reveal"><CircuitArtwork analysis={artAnalysis} circuit={artEvent?.circuit ?? 'Circuit'} /></div>
      <div className="hero-bottom landing-reveal"><span>01 / FORECAST THE GRID</span><span>SCROLL TO EXPLORE ↓</span></div>
    </section>
    <section className="event-feature frame scroll-reveal" aria-labelledby="next-event-title">
      <div className="section-kicker"><span>THE NEXT CHAPTER</span><span>01 / 03</span></div>
      <div className="feature-grid"><div><h2 id="next-event-title">{event?.name ?? 'Calendar pending'}</h2><p>{event?.circuit}, {event?.country}</p></div><div className="feature-meta"><p>{event?.sessions.find((session) => session.kind === 'R') ? <TimeLabel iso={event.sessions.find((session) => session.kind === 'R')!.start} trackZone={event.timezone} detail /> : 'Schedule pending'}</p><Link href={event ? `/${event.season}/${event.id}/` : '/2026/'}>Open race weekend <span aria-hidden="true">↗</span></Link></div></div>
    </section>
    <section className="manifesto frame scroll-reveal" aria-labelledby="manifesto-title"><div className="section-kicker"><span>THE APPROACH</span><span>02 / 03</span></div><h2 id="manifesto-title">A forecast should show its work.</h2><div className="manifesto-columns"><p>Explore the full predicted order, position ranges, and probability distributions. Every forecast carries its cutoff, model version, and input coverage.</p><p>Historical reconstructions, issued forecasts, and demonstrations are marked separately. Missing inputs appear as an explanation, never a silent gap.</p></div><Link className="text-link" href="/methodology/">Read the methodology <span aria-hidden="true">↗</span></Link></section>
    <section className="closing frame scroll-reveal"><div className="section-kicker"><span>THE EVIDENCE</span><span>03 / 03</span></div><h2>Every prediction<br />meets the result.</h2><p>Model performance is reported only after evaluated events are available.</p><Link className="action-primary" href="/performance/">View performance <span aria-hidden="true">↗</span></Link></section>
  </main>;
}

import Link from 'next/link';
import { LandingMotion } from '@/components/landing-motion';
import { LandingWeekend } from '@/components/landing-weekend';
import { getSiteData } from '@/lib/data';
import { selectCircuitOutline } from '@/lib/circuit-outline';
import { selectLandingEvent } from '@/lib/landing-event';

export default function Home() {
  const data = getSiteData();
  const initialEvent = selectLandingEvent(data.events, null).event;
  const initialArtwork = initialEvent ? {
    eventId: initialEvent.id,
    outline: selectCircuitOutline(data.events.find((event) => event.id === initialEvent.id)!, data.events, data.analyses),
  } : undefined;
  return <main id="main"><LandingMotion />
    <LandingWeekend events={data.events.map(({ id, season, name, circuit, country, timezone, sessions }) => ({ id, season, name, circuit, country, timezone, sessions }))} initialArtwork={initialArtwork} />
    <section className="manifesto frame scroll-reveal" aria-labelledby="manifesto-title"><div className="section-kicker"><span>THE APPROACH</span><span>02 / 03</span></div><h2 id="manifesto-title">A forecast should show its work.</h2><div className="manifesto-columns"><p>Explore the full predicted order, position ranges, and probability distributions. Every forecast carries its cutoff, model version, and input coverage.</p><p>Historical reconstructions, issued forecasts, and demonstrations are marked separately. Missing inputs appear as an explanation, never a silent gap.</p></div><Link className="text-link" href="/methodology/">Read the methodology <span aria-hidden="true">↗</span></Link></section>
    <section className="closing frame scroll-reveal"><div className="section-kicker"><span>THE EVIDENCE</span><span>03 / 03</span></div><h2>Every prediction<br />meets the result.</h2><p>Model performance is reported only after evaluated events are available.</p><Link className="action-primary" href="/performance/">View performance <span aria-hidden="true">↗</span></Link></section>
  </main>;
}

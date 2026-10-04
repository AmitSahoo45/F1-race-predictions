import Link from 'next/link';
import { CircuitArtwork } from '@/components/circuit-artwork';
import { LandingMotion } from '@/components/landing-motion';
import { LandingWeekend } from '@/components/landing-weekend';
import { getSiteData } from '@/lib/data';

export default function Home() {
  const data = getSiteData();
  const artAnalysis = data.analyses.find((item) => item.circuit_points.length > 2);
  const artEvent = data.events.find((item) => item.id === artAnalysis?.event_id);
  return <main id="main"><LandingMotion />
    <LandingWeekend events={data.events.map(({ id, season, name, circuit, country, timezone, sessions }) => ({ id, season, name, circuit, country, timezone, sessions }))} artwork={<CircuitArtwork analysis={artAnalysis} circuit={artEvent?.circuit ?? 'Circuit'} />} />
    <section className="manifesto frame scroll-reveal" aria-labelledby="manifesto-title"><div className="section-kicker"><span>THE APPROACH</span><span>02 / 03</span></div><h2 id="manifesto-title">A forecast should show its work.</h2><div className="manifesto-columns"><p>Explore the full predicted order, position ranges, and probability distributions. Every forecast carries its cutoff, model version, and input coverage.</p><p>Historical reconstructions, issued forecasts, and demonstrations are marked separately. Missing inputs appear as an explanation, never a silent gap.</p></div><Link className="text-link" href="/methodology/">Read the methodology <span aria-hidden="true">↗</span></Link></section>
    <section className="closing frame scroll-reveal"><div className="section-kicker"><span>THE EVIDENCE</span><span>03 / 03</span></div><h2>Every prediction<br />meets the result.</h2><p>Model performance is reported only after evaluated events are available.</p><Link className="action-primary" href="/performance/">View performance <span aria-hidden="true">↗</span></Link></section>
  </main>;
}

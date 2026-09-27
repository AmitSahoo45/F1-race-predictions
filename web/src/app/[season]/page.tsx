import type { Metadata } from 'next';
import Link from 'next/link';
import { getSiteData } from '@/lib/data';
import { TimeLabel } from '@/components/time-label';

export function generateStaticParams() {
  return [...new Set(getSiteData().events.map((event) => event.season))].map((season) => ({ season: String(season) }));
}

export function generateMetadata({ params }: { params: Promise<{ season: string }> }): Promise<Metadata> {
  return params.then(({ season }) => ({ title: `${season} Calendar` }));
}

export default async function SeasonPage({ params }: { params: Promise<{ season: string }> }) {
  const { season } = await params;
  const events = getSiteData().events.filter((event) => String(event.season) === season).sort((a, b) => a.round - b.round);
  return <main id="main" className="frame page-main"><div className="page-eyebrow">SEASON ARCHIVE / {season}</div><h1 className="page-title">The calendar<span className="accent-dot">.</span></h1><p className="page-lead">Choose a weekend to see qualifying and race forecast availability, session chronology, and the evidence behind each prediction.</p>
    <div className="calendar-head" aria-hidden="true"><span>ROUND</span><span>WEEKEND</span><span>RACE START</span><span>STATUS</span></div>
    <ol className="calendar-list">{events.map((event) => { const race = event.sessions.find((session) => session.kind === 'R'); return <li key={event.id}><Link className="calendar-row" href={`/${event.season}/${event.id}/`}><span className="round-num">{String(event.round).padStart(2, '0')}</span><span className="calendar-name"><strong>{event.name}</strong><small>{event.circuit} / {event.country}</small></span><span className="calendar-date">{race ? <TimeLabel iso={race.start} trackZone={event.timezone} /> : 'TBC'}</span><span className="calendar-status">{event.targets.every((target) => target.state === 'unavailable') ? 'Unavailable' : event.targets.some((target) => target.state === 'scheduled') ? 'Scheduled' : 'Explore'} <span aria-hidden="true">↗</span></span></Link></li>; })}</ol>
    {events.length === 0 ? <p className="empty-state">No events have been published for this season.</p> : null}
  </main>;
}

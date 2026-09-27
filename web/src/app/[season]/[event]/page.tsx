import type { Metadata } from 'next';
import { Suspense } from 'react';
import { getSiteData } from '@/lib/data';
import { WeekendDashboard } from '@/components/weekend-dashboard';
import type { DriverAnalysis } from '@/generated/site-data';

export function generateStaticParams() {
  return getSiteData().events.map((event) => ({ season: String(event.season), event: event.id }));
}

export async function generateMetadata({ params }: { params: Promise<{ season: string; event: string }> }): Promise<Metadata> {
  const { season, event: eventId } = await params;
  const event = getSiteData().events.find((item) => String(item.season) === season && item.id === eventId);
  return { title: event?.name ?? 'Race weekend', openGraph: { images: [{ url: `${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}/og/${season}-${eventId}.png`, width: 1200, height: 630, type: 'image/png' }] } };
}

export default async function EventPage({ params }: { params: Promise<{ season: string; event: string }> }) {
  const { season, event: eventId } = await params;
  const site = getSiteData();
  const event = site.events.find((item) => String(item.season) === season && item.id === eventId);
  if (!event) return <main id="main" className="frame page-main"><h1>Event not found</h1></main>;
  const analysis = site.analyses.find((item) => item.event_id === event.id);
  const leanAnalysis = analysis ? { ...analysis, drivers: analysis.drivers.map((driver: DriverAnalysis) => ({ driver_id: driver.driver_id, practice_pace_s: driver.practice_pace_s, long_run_pace_s: driver.long_run_pace_s, stints: driver.stints })) } : undefined;
  const forecasts = site.forecasts.filter((item) => item.event_id === event.id);
  return <main id="main"><Suspense fallback={<div className="frame page-main">Opening weekend…</div>}><WeekendDashboard event={event} forecasts={forecasts} analysis={leanAnalysis} /></Suspense></main>;
}

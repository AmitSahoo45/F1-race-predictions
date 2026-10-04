import type { Event } from '@/generated/site-data';

export type LandingEvent = Pick<Event, 'id' | 'season' | 'name' | 'circuit' | 'country' | 'timezone' | 'sessions'>;
type Selection = { event: LandingEvent | null; mode: 'ongoing' | 'upcoming' | 'archive' | 'calendar' };

export function selectLandingEvent(events: LandingEvent[], now: number | null): Selection {
  const dated = events.flatMap((event) => {
    if (!event.sessions.length) return [];
    const starts = event.sessions.map((session) => Date.parse(session.start));
    const ends = event.sessions.map((session) => Date.parse(session.end));
    if (![...starts, ...ends].every(Number.isFinite) || starts.some((start, index) => start >= ends[index])) return [];
    return [{ event, start: Math.min(...starts), end: Math.max(...ends) }];
  }).sort((a, b) => a.start - b.start || a.event.id.localeCompare(b.event.id));
  const latest = [...dated].sort((a, b) => b.end - a.end)[0]?.event ?? null;
  // A static server render has no current viewer clock. Avoid a build-time claim of "next".
  if (now === null || !Number.isFinite(now)) return { event: latest, mode: 'calendar' };
  const ongoing = dated.find((item) => item.start <= now && now < item.end);
  if (ongoing) return { event: ongoing.event, mode: 'ongoing' };
  const upcoming = dated.find((item) => item.start > now);
  if (upcoming) return { event: upcoming.event, mode: 'upcoming' };
  return { event: latest, mode: latest ? 'archive' : 'calendar' };
}

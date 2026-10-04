import { describe, expect, it } from 'vitest';
import { selectLandingEvent, type LandingEvent } from './landing-event';

function weekend(id: string, first: string, race: string, end: string): LandingEvent {
  return {
    id, season: 2026, name: id, circuit: 'example', country: 'Example', timezone: 'UTC',
    sessions: [
      { id: `${id}-FP1`, kind: 'FP1', start: first, end: new Date(Date.parse(first) + 3_600_000).toISOString() },
      { id: `${id}-R`, kind: 'R', start: race, end },
    ],
  };
}

const past = weekend('2026-01', '2026-03-06T09:00:00Z', '2026-03-08T09:00:00Z', '2026-03-08T11:00:00Z');
const current = weekend('2026-16', '2026-10-02T09:00:00Z', '2026-10-04T09:00:00Z', '2026-10-04T11:00:00Z');
const future = weekend('2026-17', '2026-10-09T09:00:00Z', '2026-10-11T09:00:00Z', '2026-10-11T11:00:00Z');

describe('landing weekend selection from the viewer clock', () => {
  it('selects the nearest upcoming event despite an old scheduled target or array order', () => {
    const events = [{ ...past, targets: [{ state: 'scheduled' }] }, future, current];
    expect(selectLandingEvent(events, Date.parse('2026-09-30T12:00:00Z'))).toEqual({ event: current, mode: 'upcoming' });
    expect(events.map((event) => event.id)).toEqual(['2026-01', '2026-17', '2026-16']);
  });

  it('keeps an ongoing weekend between qualifying and the race regardless of forecast state', () => {
    const ongoing = { ...current, targets: [{ state: 'unavailable' }] };
    expect(selectLandingEvent([future, past, ongoing], Date.parse('2026-10-03T23:00:00Z')))
      .toEqual({ event: ongoing, mode: 'ongoing' });
  });

  it('switches to ongoing at the first session start and remains until the final session ends', () => {
    expect(selectLandingEvent([past, current, future], Date.parse('2026-10-02T09:00:00Z')).mode).toBe('ongoing');
    expect(selectLandingEvent([past, current, future], Date.parse('2026-10-04T10:59:59Z')))
      .toEqual({ event: current, mode: 'ongoing' });
    expect(selectLandingEvent([past, current, future], Date.parse('2026-10-04T11:00:00Z')))
      .toEqual({ event: future, mode: 'upcoming' });
  });

  it('uses the most recent ended weekend with an archive mode when the static calendar ages', () => {
    expect(selectLandingEvent([future, past, current], Date.parse('2027-01-01T00:00:00Z')))
      .toEqual({ event: future, mode: 'archive' });
  });

  it('uses a neutral calendar mode before the client clock is available', () => {
    expect(selectLandingEvent([past, future, current], null)).toEqual({ event: future, mode: 'calendar' });
  });

  it('reports calendar pending without turning absent or invalid dates into a next event', () => {
    const invalid = { ...current, sessions: current.sessions.map((session) => ({ ...session, start: 'invalid' })) };
    expect(selectLandingEvent([invalid, { ...future, sessions: [] }], Date.now())).toEqual({ event: null, mode: 'calendar' });
    expect(selectLandingEvent([], Date.now())).toEqual({ event: null, mode: 'calendar' });
  });
});
